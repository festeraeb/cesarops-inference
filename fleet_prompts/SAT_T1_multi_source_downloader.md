# Task: Multi-Source Satellite Downloader (Python)

Write a complete Python script `satellite_downloader.py` that downloads satellite imagery from multiple free sources for a given bounding box and date range.

## Sources to implement (in priority order):

### 1. Element84 Earth Search STAC (NO AUTH NEEDED)
- URL: `https://earth-search.aws.element84.com/v1/search`
- Collections: `sentinel-2-l2a`, `landsat-c2-l2`
- Method: POST STAC search with bbox + datetime
- Download: COG (Cloud-Optimized GeoTIFF) direct from `assets.visual.href` or band-specific

### 2. LandsatLook STAC (NO AUTH NEEDED)
- URL: `https://landsatlook.usgs.gov/stac-server/search`
- Collections: `landsat-c2l2-sr` (surface reflectance), `landsat-c2l2-st` (surface temperature/thermal)
- Includes Landsat 4-5 TM historical + Landsat 8/9 current
- Download: GeoTIFF from asset hrefs

### 3. NASA CMR / HLS (EARTHDATA TOKEN)
- URL: `https://cmr.earthdata.nasa.gov/search/granules.json`
- Collections: `HLSS30` (Sentinel-2 30m), `HLSL30` (Landsat 30m)
- Auth: Bearer token from env var `EARTHDATA_TOKEN`
- Download: via S3 or HTTPS with auth header

### 4. ASF (Sentinel-1 SAR, EARTHDATA TOKEN)
- URL: `https://api.daac.asf.alaska.edu/services/search/param`
- Params: `platform=Sentinel-1&processingLevel=GRD_HD&bbox={W,S,E,N}&start={date}&end={date}&output=json`
- Download: via `.url` field in results, auth with Earthdata token

### 5. NASA PO.DAAC (SWOT + ICESat-2, EARTHDATA TOKEN)
- SWOT: CMR search with `short_name=SWOT_L2_HR_Raster_2.0`
- ICESat-2: CMR search with `short_name=ATL03`
- Same auth pattern as HLS

### 6. NOAA GLERL CoastWatch (Great Lakes thermal, NO AUTH)
- ERDDAP endpoint: `https://apps.glerl.noaa.gov/erddap/griddap/`
- Datasets: `glsea_anom` (surface temp anomaly), `glerl_glsea_sst` (SST)
- Download: `.nc` or `.csv` via ERDDAP query with bbox + time

### 7. GOES-16/18 ABI (SWIR + thermal, NO AUTH)
- AWS S3 bucket: `s3://noaa-goes16/ABI-L2-MCMIPC/` (CONUS)
- Bands: 7 (3.9µm SWIR), 13 (10.3µm thermal), 14 (11.2µm thermal)
- Access: public S3, no auth needed
- Alternative: Google Earth Engine catalog `NOAA/GOES/16/MCMIPM`

### 8. Sentinel-3 SLSTR (thermal + SWIR)
- Via NASA LAADS DAAC: `https://ladsweb.modaps.eosdis.nasa.gov/`
- Product: `S3A_SL_1_RBT` (Sentinel-3A SLSTR L1B)
- Auth: Earthdata token
- Bands: S5 (1.6µm SWIR), S6 (2.25µm SWIR), S7-S9 (thermal IR)

### 9. USGS Magnetic Anomaly Grid (bonus — not satellite but useful)
- URL: `https://mrdata.usgs.gov/magnetic/` (WMS/WCS)
- Or direct GeoTIFF: `https://data.usgs.gov/datacatalog/data/USGS:619a9a3ad34eb622f692f961`
- NO AUTH

## CLI interface:

```
python3 satellite_downloader.py \
  --bbox 42.9 -87.2 43.2 -86.2 \
  --dates 2026-04-01 2026-05-17 \
  --output /tmp/cesarops_downloads/ \
  --sources all \
  --cloud-max 20 \
  --bands visual,thermal,swir,sar \
  --max-scenes 50
```

Options:
- `--sources`: comma-separated list or `all`. Options: `element84,landsatlook,hls,asf,podaac,glerl,goes,sentinel3,magnetic`
- `--bands`: filter by band type: `visual,thermal,swir,sar,altimetry`
- `--cloud-max`: max cloud cover % (only applies to optical)
- `--max-scenes`: cap total downloads
- `--cloud-free-mosaic`: if set, prefer Tri-Decadal cloud-free mosaics first

## Environment variables (loaded from .env in script dir):
- `EARTHDATA_TOKEN` — for NASA CMR, HLS, ASF, PO.DAAC, LAADS
- `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` — fallback auth
- `COPERNICUS_USER` / `COPERNICUS_PASS` — for Copernicus (optional)

## Output structure:
```
/tmp/cesarops_downloads/
├── element84/
│   ├── S2A_T16TDQ_20260501_visual.tif
│   └── LC09_L2SP_20260503_B10.tif  (thermal)
├── landsatlook/
│   └── LT05_L2SP_19900815_visual.tif  (historical cloud-free)
├── hls/
│   └── HLS.S30.T16TDQ.2026121.v2.0.B04.tif
├── asf/
│   └── S1A_IW_GRDH_20260510.tif
├── glerl/
│   └── glsea_sst_20260515.nc
├── goes/
│   └── GOES16_ABI_B07_20260515_1800.nc
├── sentinel3/
│   └── S3A_SL_1_RBT_20260512_S7.tif
├── magnetic/
│   └── us_canada_mag_anomaly.tif
└── manifest.json  (summary of all downloads)
```

## manifest.json format:
```json
{
  "bbox": [42.9, -87.2, 43.2, -86.2],
  "dates": ["2026-04-01", "2026-05-17"],
  "downloads": [
    {"source": "element84", "collection": "sentinel-2-l2a", "file": "...", "date": "...", "cloud_pct": 5, "bands": ["visual"]},
    ...
  ],
  "total_files": 23,
  "total_size_mb": 450,
  "errors": ["COPERNICUS: credentials not set", ...]
}
```

## Constraints:
- Python 3.10+
- Dependencies: requests, boto3 (for GOES S3), netCDF4 (for ERDDAP/GOES .nc)
- Use `requests.Session()` with auth headers for Earthdata sources
- Graceful degradation: if a source fails (no creds, timeout, 404), log error and continue to next
- Print progress to stdout: `[element84] Searching sentinel-2-l2a... 12 scenes found`
- Cloud-free mosaic (Tri-Decadal) should be attempted FIRST if --cloud-free-mosaic is set
- Keep under 600 lines

## Output format:
Output the complete Python script as a single file.
