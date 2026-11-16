# VGM crowding model

Simulates visitors moving through the Van Gogh Museum to predict people per room per 15 minutes,
and compares scenarios (room closures, more or fewer visitors). Full explanation: project notes,
section "Crowding model approach", and Appendix B.

## Pipeline (run in order, from this folder; Python 3 with pandas, numpy, scipy, matplotlib)

| Step | Script | What it does | Runtime |
|---|---|---|---|
| 1 | `01_build_room_visits.py` | Audio-guide stops (labeled file, "listened" rule) -> room visits for a 15% sample of sessions | ~2 min |
| 2 | `02_build_arrivals.py` | Ticket scans -> visitors entering per 5 min | ~1 min |
| 3 | `03_estimate_logit.py` | Next-room / exit logit (maximum likelihood) + held-out test vs counted table | ~4 min |
| 4 | `04_stay_model.py` | Stay-time regression per room, first-room shares, lead time | <1 min |
| 5 | `05_calibrate.py` | Calibrates the "continue vs exit" constant and the exit share when the exhibition wing is closed | ~7 min |
| 6 | `06_validate.py` | Simulated vs measured crowding: normal days and the 18 May-11 Jun closure | ~10 min |
| 7 | `07_benchmark.py` | People per room: logit vs counted-table benchmark; slot table for one room (`--room R01`) | ~10 min |
| 8 | `08_build_crowding.py` | Crowding (people per room per 5 min) from ALL audio-guide sessions ("listened" rule), hourly scaling and audio-guide correction; run per month: `python 08_build_crowding.py --month 2026-08` | ~30 s/month |
| 9 | `09_crowding_descriptives.py` | Does crowding change stays, next-room choice, skipping, exit? (day fixed effects, controls) | <1 min |
| 10 | `10_weather_vs_duration.py` | Are visits longer when it rains? KNMI hourly weather (External Data) vs visit duration, with controls | <1 min |
| 11 | `11_build_tickets.py` | Tickets sold per 15-min slot (by channel) and arrival times vs booked slot, from the raw ticket file; run from the Analysis folder | ~1 min |
| 12 | `12_arrivals_model.py` | Fits and validates tickets -> people entering (`arrivals.py`: show-up rate, arrival-time spread, 09:00 door queue) | ~3 min |
| 13 | `13_mix_sensitivity.py` | Does the visitor (language) mix change crowding? Real vs typical month x weekday vs extreme mixes | ~6 min |
| 14 | `14_build_artworks.py` | People at six famous artworks per 5 min + what visitors do in those rooms (all sessions; run per month from the Analysis folder, after step 8) | ~35 s/month |
| 15 | `15_artwork_congestion.py` | Famous-artwork congestion vs people in the room; does room crowding change engagement with the artworks? | <1 min |
| 16 | `16_accuracy_data.py` | Data for the app's accuracy page: measured people per room per 15 min, and the model run on every past day (re-run when the model changes) | ~30 min |
| 17 | `17_visit_duration_nps.py` | Visit-duration file: audio guide vs no audio guide visit time; NPS vs congestion during each respondent's visit (needs `02_build_arrivals.py --start 2025-09-01 --out arrivals_5min_from2025.csv`) | <1 min |
| 18 | `18_audio_guide.py` | Audio guide vs no audio guide in the simulation: share per day and hour (`data/ag_share_hour.csv`) and the stay multiplier for visitors without a guide (`params/audio_guide.json`). Then re-run 05, 06, 07, 16 | ~2 min |

Shared code: `layout.py` (rooms, floors, adjacency - edit if the layout is wrong), `features.py` (logit factors),
`sim.py` (simulation), `arrivals.py` (tickets -> entries). Stop-to-room mapping: `station_room_mapping.csv`
(rebuild with `python build_station_mapping.py` after editing its ROOMS / STATIONS lists).

## What the repo contains, and rebuilding from raw data

**In the repo (enough to run the app and scenarios):** all code, the estimated parameters in `params/` (logit,
stay-time regression coefficients and residuals, calibration, arrivals, audio guide) and nine small data files in
`data/` (~4 MB). The app and `run_scenario.py` work straight after cloning.

**Not in the repo (VGM data, and too large):** the raw files below and everything built from them (`room_visits.csv`,
`data/crowding/`, `data/artworks/`, `output/`). Share the raw data through the team's OneDrive, never GitHub. To
re-estimate the model (e.g. change a factor in the stay-time regression in `04_stay_model.py`), get the raw data and put
it in this layout - the scripts find it relative to this folder (two levels up):

```
<project folder>/
  Raw Data/
    20260915 - Audiotour data VGM 20260201 - 20260916/   (3 CSVs)   -> step 14 (and 1, 8 with --raw)
    20260915 - Ticket sales and visitor data VGM.csv                -> steps 2, 11
    20261001 - Visit duration - IESE.xlsx                           -> step 17
  External Data/
    Amsterdam_weather_data.txt   (KNMI hourly, station 240)         -> step 10
  Analysis/
    Audiotour data - labeled/   (colleague's cleaned + labeled file, and extracts) -> steps 1, 8
    model/   <- this repo (any folder name works, but it must sit one level below the project folder)
```

**Rebuild order** (a full rebuild takes ~1.5 hours, mostly 05, 06, 07 and 16):

1. `python 01_build_room_visits.py` and `python 02_build_arrivals.py` - room visits (15% sample) and entries
   (and `python 02_build_arrivals.py --start 2025-09-01 --out arrivals_5min_from2025.csv` for step 17)
2. `python 03_estimate_logit.py`, `python 04_stay_model.py` - estimate the logit and the stay-time regression
3. `python 17_visit_duration_nps.py` - audio-guide time ratio (`params/audio_guide.json`)
4. `python 08_build_crowding.py --month 2026-02` ... `--month 2026-09` - crowding measure, one month at a time
5. `python 11_build_tickets.py`, `python 12_arrivals_model.py` - tickets -> entries for scenarios
6. `python 18_audio_guide.py` - audio-guide share per hour and stay multiplier
7. `python 05_calibrate.py` - recalibrate (always after changing the logit, the stay model or the measure)
8. `python 06_validate.py`, `python 07_benchmark.py`, `python 16_accuracy_data.py` - validation and the app's accuracy page
9. Commit the changed `params/` and `data/` files and push: the online app picks them up.

If you only change the stay-time regression: run step 4 of the table (`04_stay_model.py`), then 05, then 16 (and 06/07
if you want the validation charts). This needs `data/room_visits.csv`, i.e. step 01 and the audio-guide raw data.

## Running a scenario

```
python run_scenario.py --date 2026-08-14 --close R06
python run_scenario.py --date 2026-08-14 --close EXH --capacity 150 --reps 200
python run_scenario.py --date 2026-08-14 --arrivals-factor 1.10
```

Zones: R01-R12 (rooms 1-12), EXH (exhibition wing). Output in `output/scenarios/<name>/`.

## Folders

- `data/` - inputs built from the raw data (room_visits.csv, arrivals_5min.csv, open_rooms.csv, crowding/ ...); only the small files the app needs are in the repo
- `params/` - estimated parameters (logit.json, stay.json, entry.json, calibration.json)
- `output/` - coefficient tables, fit statistics, validation charts, scenario results


## Scenarios with ticket sales
`run_scenario.py` accepts `--tickets-factor 1.1` (sell 10% more in every slot of that day), `--tickets-per-slot 180`
(same number in every slot 09:00-16:30) or `--tickets-file slots.csv` (columns slot, tickets). Any date works: for days
without data the model uses that weekday's typical closing time and the typical language mix of that month and weekday.

## Web app (scenario tool)
`app.py` is a Streamlit app on top of the simulation: pick a day, set ticket sales (real, scaled, same number per slot,
or typed / uploaded per slot), close rooms (incl. the exhibition wing), and see expected people per room per 15 minutes
(heatmap, one chart per room vs baseline, entries) with CSV downloads. A second page, "Accuracy vs history",
shows how close the model is to measured crowding on average and for any past day.

- Run locally: `pip install -r requirements.txt`, then `streamlit run app.py` from this folder (or double-click
  `Start VGM app.bat` on Windows). It opens at http://localhost:8501. A run takes ~10-40 seconds.
- Link to a day: `http://localhost:8501/?date=2026-12-04`.
- Put online: push this folder to GitHub and deploy `app.py` on Streamlit Community Cloud (free) or any server.
  Files the app needs: the .py files, `params/`, `.streamlit/`, and in `data/` only arrivals_5min.csv,
  tickets_slot.csv, open_rooms.csv, closing_times.csv, lang_by_date.csv, measured_15min.csv.gz,
  simulated_15min.csv.gz, measured_days.csv and ag_share_hour.csv (~4 MB; `.gitignore` already keeps everything else out). room_visits.csv and the raw data
  are NOT needed (and should not be uploaded).

## v1.4: labeled audio-tour data, "listened" rule (16 Nov 2026)
A colleague cleaned the audio-tour export (duplicates collapsed) and labeled each stop with room-presence evidence
("label level 1": Strong / Moderate / Weak / Unresolved, rules v1.0). Steps 01 and 08 now use it with the rule:
- drop "Selected Only" stops (tapped, not listened - all of them are labeled Weak);
- drop Unresolved stops, except in the exhibition wing, where every listened stop is labeled Unresolved (its
  stations "Vincent", "Theo", "Jo", "Prologue"... are mapped to the wing in station_room_mapping.csv).

Keeps ~70% of stops and 99.3% of sessions (rooms seen per visitor 9.6, was 12.4). Strong-only (or Strong + Moderate)
was rejected: the exhibition wing and Room 6 have no Strong or Moderate stops, and paths lose half their rooms.
`--raw` on 01 and 08 gives the old all-stops build. Old version kept in `archive/2026-11-16_v1.3_all_stops/` (not in git).

The labeled file is 2.5 GB and too slow to read per month, so it is cut into extracts once (12 byte-range passes):
`Audiotour data - labeled/listened_sample15.csv.gz` (15% sample, for 01: `python 01_build_room_visits.py --file
"../Audiotour data - labeled/listened_sample15.csv.gz"`) and `listened_by_month/` (all sessions, for 08).
Then: 03, 04, 18 (stay multiplier; the audio-guide share per hour is kept from the all-stops build, since it is a fact
about visitors), 05, 06, 07, 16. `SAMPLE_RATE` in 06 = 0.1365 (sessions with 3+ rooms in the sample / all).

Results (people per room, 10:00-17:00, each model vs the measure built with its own rule):
| | v1.3 all stops | v1.4 listened |
|---|---|---|
| Normal days: correlation / avg error (cells >= 20 people) | 0.888 / 18.1% | 0.889 / 20.2% |
| Closure 18 May-11 Jun (out of sample): correlation / avg error | 0.947 / 15.8% | 0.934 / 16.0% |
| Next-room logit, held-out top-1 accuracy | 60% | 72% |
| Calibration: continue_shift / exit share with wing closed | 0.21 / 0.14 | 0.18 / 0.23 |
On the new measure the old model does worse (0.885 / 21.3%; closure 0.913 / 18.1%). The new rule lowers measured
crowding in Rooms 10 (-23%), 8 (-15%), 1 (-14%) and raises Rooms 6 and 11 (+14%). Remaining biases: exhibition wing
+13%, Room 11 -13%, other rooms within +-7%.

## Audio guide correction (27 Oct 2026)
Visitors without an audio guide spend ~12% less time in the museum (visit-duration file, step 17). Order to rebuild:
17 (writes `non_ag_time_ratio`) -> 08 per month (crowding measure: weight = 1 + (factor - 1) x ratio) -> 18 (share per
hour + stay multiplier for non-guide visitors in the simulation) -> 05 (recalibrate) -> 06, 07 -> 16 (app accuracy data).
Scenario lever: `run_scenario.py --ag-share 0.6`, or "Change audio-guide take-up" in the app.

## Capacity check (experience-neutral capacity, 12 Nov 2026)
`capacity.py` + the "Capacity check" tab in the app. Assumes today's sold-out days (>= 5,000 tickets) are VGM's acceptable
crowding. Each room's limit = P75/P90/P95 of its SIMULATED 15-min crowding, 10:00-17:00, on sold-out days when open
(simulated, so the model's peak under-estimate cancels). Crowding over the limit is counted in PERSON-HOURS (people
over the limit x time: 1 person over for an hour = 1, 30 people over for an hour = 30), so being one person over barely
counts. A scenario passes if, in every open room, its person-hours over the limit are no more than today's + an
allowance of extra person-hours set with a slider (0-20, default 1 = one person over for an hour; 0 = no worse than
today). "Find the ticket level" scales tickets (all slots or a time window) by bisection, 10 simulation runs per try
(~1 minute, about +-3%). Example (14 Aug 2026, P90), share of today's tickets allowed with 0 / 1 / 3 / 5 / 10 extra
person-hours (model v1.4): closing Room 5 -> 94% / 97% / 97% / 98% / 100%; closing the exhibition wing -> 89% / 91% / 94% / 94% / 98%.
