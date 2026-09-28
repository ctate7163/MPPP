# Scapes used

Made by `scripts/sites_table.py` from the MPPP manifests of the scape folders on disk (the v0p15 manifests are the earlier processing of the same image sets; the v0p22 manifests are the runs analysed in notebook 04). Span is the largest horizontal distance between station centres; stations are (site, drive) pairs. LMST is per image, from the label (`LOCAL_MEAN_SOLAR_TIME`). Histogram: `sites_lmst.png`.

| scape | cameras | manifest | site | sols | n_sols | images | navcam (left) | Mastcam-Z | Navcam scales | stations | span m | LMST h | LMST spread h | sun el. deg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Three Forks | Navcam | v0p15 | 32 | 684–693 | 4 | 52 | 52 (28) | 0 | 0.25/0.5/1 | 3 | 3.1 | 12.3–15.8 | 3.5 | 40–79 |
| Three Forks + Z34 | Navcam + Mastcam-Z 34 | v0p22 | 32 | 684–693 | 7 | 401 | 59 (35) | 342 | 0.25/0.5/1 | 3 | 3.1 | 10.9–15.8 | 4.9 | 40–79 |
| Rockytop | Navcam | v0p15 | 26 | 461–509 | 22 | 168 | 168 (102) | 0 | 0.25/0.5/1 | 9 | 21.6 | 8.7–17.0 | 8.2 | 8–47 |
| Rockytop + Z34 | Navcam + Mastcam-Z 34 | v0p15 | 26 | 461–530 | 38 | 1020 | 194 (122) | 826 | 0.25/0.5/1 | 9 | 21.6 | 8.7–17.4 | 8.7 | 2–49 |
| Belva | Navcam | v0p15 | 37-38-39 | 762–815 | 31 | 298 | 298 (198) | 0 | 0.25/0.5/1 | 13 | 704.4 | 6.9–17.8 | 11.0 | 10–86 |
| Airey Hill + Z34 | Navcam + Mastcam-Z 34 | v0p22 | 47 | 961–991 | 8 | 230 | 78 (60) | 152 | 0.25/0.5/1 | 3 | 5.9 | 5.7–16.6 | 10.9 | 7–85 |
| Bell Island | Navcam | v0p15 | 70-71 | 1451–1466 | 11 | 145 | 145 (87) | 0 | 0.25/0.5/1 | 6 | 8.4 | 8.7–18.0 | 9.3 | 7–86 |
| Bell Island + Z34 | Navcam + Mastcam-Z 34 | v0p22 | 70-71 | 1451–1467 | 12 | 167 | 159 (101) | 8 | 0.25/0.5/1 | 6 | 8.5 | 8.7–18.0 | 9.3 | 7–87 |
| Taylor Fjellet | Navcam | v0p15 | 78-79 | 1601–1645 | 31 | 365 | 365 (245) | 0 | 0.25/0.5/1 | 11 | 37.4 | 6.7–21.6 | 14.9 | -48–87 |

## By site

Each site is the union of its scapes, each image counted once.

| site | rover site | sols | sols with images | Navcam (left) | Mastcam-Z 34 | stations | span m | LMST h | Navcam LMST IQR h | sun el. deg |
|---|---|---|---|---|---|---|---|---|---|---|
| Three Forks | 32 | 684–693 | 7 | 60 (36) | 342 | 3 | 3.1 | 10.9–15.8 | 0.7 | 40–79 |
| Rockytop | 26 | 461–530 | 38 | 194 (122) | 826 | 9 | 21.6 | 8.7–17.4 | 2.1 | 2–49 |
| Belva | 37-38-39 | 762–815 | 31 | 298 (198) | 0 | 13 | 704.4 | 6.9–17.8 | 2.2 | 10–86 |
| Airey Hill | 47 | 961–991 | 8 | 78 (60) | 152 | 3 | 5.9 | 5.7–16.6 | 6.6 | 7–85 |
| Bell Island | 70-71 | 1451–1467 | 12 | 159 (101) | 8 | 6 | 8.5 | 8.7–18.0 | 1.9 | 7–87 |
| Taylor Fjellet | 78-79 | 1601–1645 | 31 | 365 (245) | 0 | 11 | 37.4 | 6.7–21.6 | 2.4 | -48–87 |
