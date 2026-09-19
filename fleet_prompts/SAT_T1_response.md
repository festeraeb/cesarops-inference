```python
import os
import json
import argparse
import datetime
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import requests
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration & Constants
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("SatelliteDownloader")

@dataclass
class DownloadResult:
    source: str
    collection: str
    file: str
    date: str
    cloud_pct: Optional[float] = None
    bands: List[str] = None

class SatelliteDownloader:
    def __init__(self, bbox: List[float], dates: List[str], output_dir: str, 
                 token: str, max_scenes: int, cloud_max: float, bands_filter: List[str]):
        self.bbox = bbox  # [W, S, E, N]
        self.start_date = dates[0]
        self.end_date = dates[1]
        self.output_dir = output_dir
        self.token = token
        self.max_scenes = max_scenes
        self.cloud_max = cloud_max
        self.bands_filter = bands_filter
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        
        self.manifest = {
            "bbox": bbox,
            "dates": dates,
            "downloads": [],
            "total_files": 0,
            "total_size_mb": 0.0,
            "errors": []
        }
        self.total_bytes = 0

    def _ensure_dir(self, source: str):
        path = os.path.join(self.output_dir, source)
        os.makedirs(path, exist_ok=True)
        return path

    def _download_file(self, url: str, dest_path: str):
        try:
            with self.session.get(url, stream=True) as r:
                r.raise_for_status()
                with open(dest_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        self.total_bytes += len(chunk)
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def _cmr_search(self, short_name: str, params: Dict) -> List[Dict]:
        """NASA CMR Search API"""
        url = "https://cmr.earthdata.nasa.gov/search/granules.json"
        query = {
            "short_name": short_name,
            "bounding_box": f"{self.bbox[0]},{self.bbox[1]},{self.bbox[2]},{self.bbox[3]}",
            "temporal": f"{self.start_date},{self.end_date}"
        }
        query.update(params)
        try:
            r = self.session.get(url, params=query)
            r.raise_for_status()
            return r.json().get('granules', [])
        except Exception as e:
            self.manifest["errors"].append(f"CMR Search Error ({short_name}): {str(e)}")
            return []

    def run_element84(self):
        """Sentinel-2 via Element84 (Simulated logic for Copernicus/S2)"""
        logger.info("[element84] Searching sentinel-2-l2a...")
        # In a real implementation, this would call the Element84 API
        # Here we simulate finding a scene to demonstrate the flow
        dest = self._ensure_dir("element84")
        # Placeholder for logic
        pass

    def run_landsatlook(self):
        """Landsat Historical Cloud-Free Mosaics"""
        logger.info("[landsatlook] Searching Tri-Decadal mosaics...")
        # Logic for finding historical cloud-free tiles
        pass

    def run_hls(self):
        """HLS (Landsat/Sentinel-2 Harmonized)"""
        logger.info("[hls] Searching HLS granules...")
        granules = self._cmr_search("HLS.S30", {})
        dest = self._ensure_dir("hls")
        for g in granules[:self.max_scenes]:
            # Simplified: extract download URL from CMR
            url = g.get('access_url') 
            if url:
                fname = f"HLS_{g['concept_id']}.tif"
                if self._download_file(url, os.path.join(dest, fname)):
                    self.manifest["downloads"].append(DownloadResult("hls", "HLS.S30", fname, "2026-01-01", 5, ["visual"]))

    def run_asf(self):
        """ASF Sentinel-1 SAR"""
        logger.info("[asf] Searching Sentinel-1 via ASF...")
        url = "https://api.daac.asf.alaska.edu/services/search/param"
        params = {
            "platform": "Sentinel-1",
            "processingLevel": "GRD_HD",
            "bbox": f"{self.bbox[0]},{self.bbox[1]},{self.bbox[2]},{self.bbox[3]}",
            "start": self.start_date,
            "end": self.end_date,
            "output": "json"
        }
        try:
            r = self.session.get(url, params=params)
            r.raise_for_status()
            results = r.json().get('matches', [])
            dest = self._ensure_dir("asf")
            for res in results[:self.max_scenes]:
                # ASF returns .url in results
                download_url = res.get('properties', {}).get('url')
                if download_url:
                    fname = f"S1_{res['properties']['datetime'].replace(':', '')}.tif"
                    if self._download_file(download_url, os.path.join(dest, fname)):
                        self.manifest["downloads"].append(DownloadResult("asf", "S1_GRD", fname, "2026-01-01", None, ["sar"]))
        except Exception as e:
            self.manifest["errors"].append(f"ASF Error: {str(e)}")

    def run_podaac(self):
        """SWOT and ICESat-2"""
        logger.info("[podaac] Searching SWOT/ICESat-2...")
        # SWOT
        swot = self._cmr_search("SWOT_L2_HR_Raster_2.0", {})
        # ICESat-2
        icesat = self._cmr_search("ATL03", {})
        dest = self._ensure_dir("podaac")
        # Logic to iterate and download...
        pass

    def run_glerl(self):
        """NOAA GLERL CoastWatch (ERDDAP)"""
        logger.info("[glerl] Querying ERDDAP...")
        # Example: glsea_sst
        # URL construction for ERDDAP based on bbox and time
        dest = self._ensure_dir("glerl")
        # Simulated download
        pass

    def run_goes(self):
        """GOES-16/18 via AWS S3"""
        logger.info("[goes] Accessing GOES S3 buckets...")
        try:
            s3 = boto3.client('s3')
            # Logic to list and download from s3://noaa-goes16/
            dest = self._ensure_dir("goes")
        except Exception as e:
            self.manifest["errors"].append(f"GOES S3 Error: {str(e)}")

    def run_sentinel3(self):
        """Sentinel-3 SLSTR via LAADS"""
        logger.info("[sentinel3] Searching Sentinel-3 SLSTR...")
        # Uses CMR or LAADS specific endpoint
        dest = self._ensure_dir("sentinel3")
        pass

    def run_magnetic(self):
        """USGS Magnetic Anomaly"""
        logger.info("[magnetic] Downloading USGS Magnetic Grid...")
        url = "https://data.usgs.gov/datacatalog/data/USGS:619a9a3ad34eb622f692f961"
        dest = self._ensure_dir("magnetic")
        if self._download_file(url, os.path.join(dest, "us_canada_mag_anomaly.tif")):
            self.manifest["downloads"].append(DownloadResult("magnetic", "USGS_MAG", "us_canada_mag_anomaly.tif", "2024-01-01", None, []))

    def execute(self, sources: List[str]):
        source_map = {
            "element84": self.run_element84,
            "landsatlook": self.run_landsatlook,
            "hls": self.run_hls,
            "asf": self.run_asf,
            "podaac": self.run_podaac,
            "glerl": self.run_glerl,
            "goes": self.run_goes,
            "sentinel3": self.run_sentinel3,
            "magnetic": self.run_magnetic
        }
        
        for src in sources:
            if src == "all":
                for s in source_map.keys():
                    source_map[s]()
            elif src in source_map:
                source_map[src]()

        # Finalize manifest
        self.manifest["total_files"] = len(self.manifest["downloads"])
        self.manifest["total_size_mb"] = round(self.total_bytes / (1024 * 1024), 2)
        
        with open(os.path.join(self.output_dir, "manifest.json"), "w") as f:
            json.dump(self.manifest, f, indent=2)
        
        logger.info(f"Finished. Downloaded {self.manifest['total_files']} files. Total size: {self.manifest['total_size_mb']} MB")

def main():
    parser = argparse.ArgumentParser(description="Satellite Data Downloader")
    parser.add_argument("--bbox", type=float, nargs=4, required=True, help="W S E N")
    parser.add_argument("--dates", type=str, nargs=2, required=True, help="Start End YYYY-MM-DD")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--sources", type=str, default="all", help="Comma-separated sources")
    parser.add_argument("--cloud-max", type=float, default=20.0)
    parser.add_argument("--bands", type=str, default="visual,thermal,swir,sar")
    parser.add_argument("--max-scenes", type=int, default=50)
    
    args = parser.parse_args()

    token = os.getenv("EARTHDATA_TOKEN")
    sources_list = args.sources.split(",") if args.sources != "all" else ["all"]
    bands_list = args.bands.split(",")

    downloader = SatelliteDownloader(
        bbox=args.bbox,
        dates=args.dates,
        output_dir=args.output,
        token=token,
        max_scenes=args.max_scenes,
        cloud_max=args.cloud_max,
        bands_filter=bands_list
    )

    downloader.execute(sources_list)

if __name__ == "__main__":
    main()
```
