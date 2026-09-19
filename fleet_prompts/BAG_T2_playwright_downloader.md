# Task: Playwright BAG File Downloader from NCEI

Write a Python script that uses Playwright to download BAG files from NOAA's NCEI Bathymetric Data Viewer for a given bounding box.

## Why Playwright:
NCEI's data viewer (https://www.ncei.noaa.gov/maps/bathymetry/) is JavaScript-driven. Direct URL scraping doesn't work — you need a headless browser to:
1. Navigate to the viewer
2. Set the bounding box
3. Select NOS Hydrographic Survey data
4. Click through to download BAG files

## Script: `download_bag_files.py`

### CLI:
```
python3 download_bag_files.py \
  --bbox 45.78 -84.85 45.88 -84.60 \
  --output /data/cesarops/bathymetry/straits_mackinac/ \
  --max-files 20
```

### Flow:
1. Launch headless Chromium via Playwright
2. Navigate to `https://www.ncei.noaa.gov/maps/bathymetry/`
3. Wait for map to load
4. Set the bounding box (zoom to area, or use the search/filter UI)
5. Find all available NOS survey BAG files in the area
6. For each file: click download, save to output dir
7. Print manifest of downloaded files

### Alternative approach (if the map viewer is too complex):
Use the NCEI data access API directly:
- `https://www.ngdc.noaa.gov/mgg/bathymetry/hydro.html` has a search form
- Or use the NOS survey catalog: `https://www.ncei.noaa.gov/products/nos-hydrographic-survey`
- The actual BAG files are often at URLs like: `https://data.ngdc.noaa.gov/platforms/ocean/nos/coast/H12001-H14000/H13607/BAG/H13607_MB_50cm_LWD_1of1.bag`

If direct URLs can be constructed from survey IDs, skip Playwright and just use requests. Only use Playwright if the download requires JavaScript interaction.

### Also implement:
A survey ID lookup function that queries NCEI for available surveys in a bbox:
```python
def find_surveys_in_bbox(lat_min, lon_min, lat_max, lon_max) -> List[str]:
    """Returns list of NOS survey IDs (e.g., H13607) covering the bbox."""
```

### Dependencies:
- playwright (pip install playwright; playwright install chromium)
- requests (fallback for direct downloads)
- argparse

### Output:
Complete Python script, under 200 lines. Prefer direct HTTP downloads over Playwright if possible — only use the browser as last resort for JavaScript-gated content.
