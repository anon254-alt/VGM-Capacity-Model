# VGM crowding model

Simulates visitors moving through the Van Gogh Museum to predict people per room per 15 minutes,
and compares scenarios (room closures, more or fewer visitors). Full explanation: project notes,
section "Crowding model approach", and Appendix B.

## Pipeline (run in order, from this folder; Python 3 with pandas, numpy, scipy, matplotlib)

| Step | Script | What it does | Runtime |
|---|---|---|---|
| 1 | `01_build_room_visits.py` | Audio-guide stops -> room visits for a 15% sample of sessions | ~2 min |
| 2 | `02_build_arrivals.py` | Ticket scans -> visitors entering per 5 min | ~1 min |
| 3 | `03_estimate_logit.py` | Next-room / exit logit (maximum likelihood) + held-out test vs counted table | ~4 min |
| 4 | `04_stay_model.py` | Stay-time regression per room, first-room shares, lead time | <1 min |
| 5 | `05_calibrate.py` | Calibrates the "continue vs exit" constant and the exit share when the exhibition wing is closed | ~7 min |
| 6 | `06_validate.py` | Simulated vs measured crowding: normal days and the 18 May-11 Jun closure | ~10 min |
| 7 | `07_benchmark.py` | People per room: logit vs counted-table benchmark; slot table for one room (`--room R01`) | ~10 min |
| 8 | `08_build_crowding.py` | Crowding (people per room per 5 min) from ALL audio-guide sessions, hourly scaling; run per month from the Analysis folder: `python model/08_build_crowding.py --month 2026-08` | ~30 s/month |
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

Steps 1-2 must run where the raw data is (the Proyecto Ops folder). Shared code: `layout.py` (rooms,
floors, adjacency - edit if the layout is wrong), `features.py` (logit factors), `sim.py` (simulation).

## Running a scenario

```
python run_scenario.py --date 2026-08-14 --close R06
python run_scenario.py --date 2026-08-14 --close EXH --capacity 150 --reps 200
python run_scenario.py --date 2026-08-14 --arrivals-factor 1.10
```

Zones: R01-R12 (rooms 1-12), EXH (exhibition wing). Output in `output/scenarios/<name>/`.

## Folders

- `data/` - inputs built by steps 1-2 (room_visits.csv, arrivals_5min.csv, open_rooms.csv)
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

## Audio guide correction (27 Oct 2026)
Visitors without an audio guide spend ~12% less time in the museum (visit-duration file, step 17). Order to rebuild:
17 (writes `non_ag_time_ratio`) -> 08 per month (crowding measure: weight = 1 + (factor - 1) x ratio) -> 18 (share per
hour + stay multiplier for non-guide visitors in the simulation) -> 05 (recalibrate) -> 06, 07 -> 16 (app accuracy data).
Scenario lever: `run_scenario.py --ag-share 0.6`, or "Change audio-guide take-up" in the app.
