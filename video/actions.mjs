// RoomEQ walkthrough: one action per narration segment (see script.json).
// Recorded against `roomeq demo` (simulated room + virtual phone) on a spare port.
const LAN = process.env.ROOMEQ_LAN || "http://192.168.1.2:3200";

export default (h) => {
  const { page } = h;
  const $ = (sel) => page.locator(sel);
  const jobDone = (label) => $("#job-state").filter({ hasText: `${label}: ✓ Done` }).waitFor({ timeout: 180_000 });

  return {
    async intro() {
      await h.point($(".brand h1"), { hover: false });
      await h.cue("measures the room");
      await h.point($("#chart-resp"), { hover: false });
    },

    async demo_mode() {
      await h.cue("blue badge");
      await h.point($("#demo-pill"));
      await h.cue("virtual phone");
      await h.point($("#preset-name"), { hover: false });
    },

    async controls() {
      await h.cue("E Q on");
      await h.point($("#eq-on"));
      await h.cue("or bypass");
      await h.click($("#eq-off"));
      await h.cue("compare tone");
      await h.click($("#eq-on"));
      await h.cue("volume trim");
      await h.point($("#volume"));
      await h.cue("live meters");
      await h.point($("#m-in"));
      await h.cue("the limiter");
      await h.point($("#s-gr"));
      await h.cue("added latency");
      await h.point($("#s-lat"));
      await h.cue("clock drift");
      await h.point($("#s-drift"));
      await h.cue("glitch counter");
      await h.point($("#s-glitch"));
    },

    async safety() {
      await h.cue("panic button");
      await h.click($("#panic"));
      await h.cue("Press it again");
      await h.click($("#panic"));
      await h.cue("look-ahead limiter");
      await h.point($("#s-gr"));
    },

    async measure_card() {
      await h.scrollTo(page.getByRole("heading", { name: "Measure", exact: true }), "center");
      await h.cue("scan this card's code");
      await h.point($("#phone-status"));
      await h.cue("virtual phone is already connected");
      await h.point($("#phone-status .pill"));
    },

    async phone_page() {
      // what a phone sees on its first visit: the plain-HTTP page with the certificate steps
      await page.goto(`${LAN}/`);
      await page.waitForTimeout(1200);
      await h.cue("trusting");
      await h.point($("#insecure ol li").first());
      await h.cue("three short steps");
      await h.point($("#insecure ol li").nth(2));
      await h.cue("a start button");
      await h.point($("#start"));
    },

    async autotune_start() {
      await h.goto("/");
      await page.waitForTimeout(800);
      await h.scrollTo(page.getByRole("heading", { name: "Measure", exact: true }), "start");
      await h.cue("Auto-tune");
      await h.click($("#start-autotune"));
      await h.cue("sweep");
      await h.point($("#job-log"));
    },

    async autotune_rounds() {
      await h.cue("measured twice");
      await h.point($("#job-log"));
      await jobDone("Auto-tune");
      await page.waitForTimeout(800);
    },

    async result_chart() {
      await h.scrollTo($(".chart-card"), "start");
      await h.cue("Orange");
      await h.point($("#legend .key").nth(0));
      await h.cue("Blue is after");
      await h.point($("#legend .key").nth(1));
      await h.cue("dotted line");
      await h.point($("#legend .key").nth(2));
      await h.cue("The bands");
      await h.point($("#chart-resp"));
    },

    async eq_curve() {
      await h.scrollTo($("#chart-eq"), "center");
      await h.cue("Hover anywhere");
      await h.point($("#chart-eq"));
      await h.cue("as a table");
      await h.click(page.getByText("Show as table"));
      await page.waitForTimeout(1500);
      await h.click(page.getByText("Show as table"));
    },

    async filters() {
      await h.scrollTo(page.getByRole("heading", { name: "Filters", exact: true }), "start");
      await h.cue("Frequency, gain and Q");
      const gain = $("#filter-rows tr").first().locator("input").nth(1);
      await h.point(gain);
      await h.cue("soften this one");
      await gain.fill("");
      await h.typeInto(gain, "-4", 120);
      await h.cue("and apply");
      await h.click($("#apply-filters"));
      await h.cue("dashed prediction");
      await h.scrollTo($(".chart-card"), "start");
      await h.point($("#chart-resp"), { hover: false });
    },

    async verify() {
      await h.scrollTo(page.getByRole("heading", { name: "Measure", exact: true }), "start");
      await h.cue("Verify E Q");
      await h.click($("#start-verify"));
      await h.cue("Four sweeps");
      await h.point($("#job-log"));
      await jobDone("Verify");
      await page.waitForTimeout(600);
      await h.point($("#job-result"));
    },

    async real_room() {
      await h.scrollTo(page.getByRole("heading", { name: "Presets", exact: true }), "center");
      await h.cue("a real result");
      await h.select($("#preset-select"), { label: "Living room (measured)" });
      await h.click($("#preset-load"));
      await page.waitForTimeout(700);
      await h.scrollTo($(".chart-card"), "start");
      await h.cue("Before");
      await h.point($("#legend .key").nth(0));
      await h.cue("After, measured");
      await h.point($("#legend .key").nth(1));
      await h.cue("saved as a preset");
      await h.point($("#preset-name"), { hover: false });
    },

    async notes() {
      await h.scrollTo(page.getByRole("heading", { name: "Notes", exact: true }), "center");
      await h.cue("These notes");
      await h.point($("#notes li").first());
      await h.cue("room null");
      await h.point($("#notes li").last());
      await h.cue("turn the subwoofer down");
      await h.point($("#limits"));
    },

    async outro() {
      await h.scrollTop();
      await h.cue("allocation free");
      await h.point($("#s-glitch"));
      await h.cue("RoomEQ. Measure");
      await h.click($("#theme"));
      await page.waitForTimeout(400);
      await h.click($("#theme"));
      await h.point($(".brand h1"), { hover: false });
    },
  };
};
