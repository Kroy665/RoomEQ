"""Local certificate authority + server certificate so the phone gets a real secure context.

iPhone Safari only allows microphone access on HTTPS, and it does not apply a "visit anyway" exception
to WebSocket connections. So RoomEQ creates its own small root CA (installed once on the phone)
and signs a short-lived server certificate for the Mac's LAN addresses with it. The server
certificate follows Apple's TLS rules: SAN present, serverAuth EKU, at most 825 days.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from ..presets import home


@dataclass(frozen=True)
class CertPaths:
    ca_pem: Path
    ca_der: Path
    cert: Path
    key: Path


def lan_ips() -> list[str]:
    ips: list[str] = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))          # no packet is sent
        ip = s.getsockname()[0]
        s.close()
        if not ip.startswith(("127.", "169.254.", "0.")):
            ips.append(ip)
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips and not ip.startswith(("127.", "169.254.")):
                ips.append(ip)
    except OSError:
        pass
    return ips


def mdns_name() -> str:
    h = socket.gethostname()
    return h if h.endswith(".local") else f"{h.split('.')[0]}.local"


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _write_key(key: ec.EllipticCurvePrivateKey, path: Path) -> None:
    path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                       serialization.NoEncryption()))
    path.chmod(0o600)


def _make_ca(paths: CertPaths, ca_key_path: Path) -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"RoomEQ Local CA ({socket.gethostname()})"),
                      x509.NameAttribute(NameOID.ORGANIZATION_NAME, "RoomEQ")])
    ski = x509.SubjectKeyIdentifier.from_public_key(key.public_key())
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(_now() - dt.timedelta(days=1)).not_valid_after(_now() + dt.timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                         content_commitment=False, key_encipherment=False, data_encipherment=False,
                                         key_agreement=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(ski, critical=False)
            .sign(key, hashes.SHA256()))
    _write_key(key, ca_key_path)
    paths.ca_pem.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    paths.ca_der.write_bytes(cert.public_bytes(serialization.Encoding.DER))


def _server_cert_ok(path: Path, hosts: list[str], ips: list[str]) -> bool:
    if not path.exists():
        return False
    cert = x509.load_pem_x509_certificate(path.read_bytes())
    if cert.not_valid_after_utc < _now() + dt.timedelta(days=30):
        return False
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    have_ips = {str(i) for i in san.get_values_for_type(x509.IPAddress)}
    have_dns = set(san.get_values_for_type(x509.DNSName))
    return set(ips) <= have_ips and set(hosts) <= have_dns


def _make_server(paths: CertPaths, ca_key_path: Path, hosts: list[str], ips: list[str]) -> None:
    ca_key = serialization.load_pem_private_key(ca_key_path.read_bytes(), password=None)
    ca = x509.load_pem_x509_certificate(paths.ca_pem.read_bytes())
    key = ec.generate_private_key(ec.SECP256R1())
    san = [x509.DNSName(h) for h in hosts] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    cert = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hosts[0])]))
            .issuer_name(ca.subject).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(_now() - dt.timedelta(days=1)).not_valid_after(_now() + dt.timedelta(days=397))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_encipherment=False, content_commitment=False,
                                         data_encipherment=False, key_agreement=True, key_cert_sign=False,
                                         crl_sign=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    _write_key(key, paths.key)
    paths.cert.write_bytes(cert.public_bytes(serialization.Encoding.PEM) + paths.ca_pem.read_bytes())


def ensure_certs(directory: Path | None = None, extra_hosts: list[str] | None = None) -> CertPaths:
    """Create the CA once; (re)issue the server certificate when the Mac's IP changes or it nears expiry."""
    d = directory or home() / "certs"
    d.mkdir(parents=True, exist_ok=True)
    paths = CertPaths(d / "roomeq-ca.pem", d / "roomeq-ca.cer", d / "server.pem", d / "server-key.pem")
    ca_key = d / "roomeq-ca-key.pem"
    if not (paths.ca_pem.exists() and ca_key.exists()):
        _make_ca(paths, ca_key)
        for p in (paths.cert, paths.key):
            p.unlink(missing_ok=True)
    hosts = [mdns_name(), "localhost"] + (extra_hosts or [])
    ips = lan_ips() + ["127.0.0.1"]
    if not _server_cert_ok(paths.cert, hosts, ips):
        _make_server(paths, ca_key, hosts, ips)
    return paths
