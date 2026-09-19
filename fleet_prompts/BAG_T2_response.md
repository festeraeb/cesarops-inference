

```python
import argparse
import os
import sys
import time
import requests
from typing import List
from playwright.sync_api import sync_playwright

def find_surveys_in_bbox(lat_min, lon_min, lat_max, lon_max) -> List[str]:
    """
    Queries NCEI NOS Hydrographic Survey catalog for surveys in bbox.
    Returns list of survey IDs (e.g., H13607).
    """
    # NCEI NOS Survey Catalog API endpoint
    # We use the search endpoint to find surveys overlapping the bbox
    url = "https://www.ncei.noaa.gov/access/metadata/portal/api/v1/survey"
    params = {
        "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "format": "json",
        "limit": 100
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # The API might return different structures; let's handle common cases
        surveys = []
        if isinstance(data, list):
            for item in data:
                survey_id = item.get('surveyId') or item.get('id') or item.get('survey_id')
                if survey_id:
                    surveys.append(survey_id)
        elif isinstance(data, dict):
            results = data.get('results', data.get('data', []))
            if isinstance(results, list):
                for item in results:
                    survey_id = item.get('surveyId') or item.get('id') or item.get('survey_id')
                    if survey_id:
                        surveys.append(survey_id)
        
        # Deduplicate
        return list(set(surveys))
    except Exception as e:
        print(f"Warning: API lookup failed ({e}), falling back to Playwright.")
        return []

def download_via_playwright(bbox, output_dir, max_files):
    """
    Uses Playwright to interact with the NCEI Bathymetric Data Viewer
    to find and download BAG files.
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = context.new_page()
        
        try:
            print(f"[*] Navigating to NCEI Bathymetry Viewer...")
            page.goto("https://www.ncei.noaa.gov/maps/bathymetry/", wait_until="networkidle", timeout=60000)
            
            # Wait for map to load
            print("[*] Waiting for map to load...")
            page.wait_for_selector("#map", timeout=30000)
            
            # Try to set the bounding box using the search/filter UI if available
            # The viewer often has a search box or coordinate input
            # Attempt to find an input field for coordinates or search
            search_input = page.query_selector("input[type='text'], input[placeholder*='search'], input[placeholder*='coordinate']")
            if search_input:
                print("[*] Setting bounding box in search UI...")
                # Format bbox as lat,lon,lat,lon or similar depending on UI
                bbox_str = f"{lat_min},{lon_min},{lat_max},{lon_max}"
                search_input.fill(bbox_str)
                search_input.press("Enter")
                time.sleep(5)
            else:
                print("[*] No direct search input found. Attempting to zoom to bbox via map interaction...")
                # If no search input, we might need to click and drag or use other mechanisms.
                # For simplicity in this script, we rely on the API fallback or assume the map centers.
                # A more robust solution would involve finding the specific map control.
                # Here we just wait and hope the default view or previous state helps, 
                # but primarily we rely on the API for IDs and direct download.
                pass

            # Look for download links or survey results
            # This part is highly dependent on the current DOM structure of the viewer.
            # Since the DOM changes, we'll try to find links containing 'bag' or 'download'
            print("[*] Scanning for BAG file links...")
            links = page.query_selector_all("a[href*='bag'], a[href*='BAG'], a[href*='download']")
            
            downloaded_files = []
            for link in links[:max_files]:
                href = link.get_attribute("href")
                if href and href.startswith("http"):
                    filename = href.split("/")[-1]
                    if filename.lower().endswith(".bag"):
                        file_path = os.path.join(output_dir, filename)
                        if not os.path.exists(file_path):
                            print(f"[*] Downloading {filename}...")
                            try:
                                # Playwright can download files directly
                                with page.expect_download() as download_info:
                                    link.click()
                                download = download_info.value
                                download.save_as(file_path)
                                downloaded_files.append(file_path)
                                print(f"[+] Saved: {file_path}")
                            except Exception as e:
                                print(f"[!] Failed to download {filename}: {e}")
                                # Fallback to requests if Playwright download fails
                                try:
                                    resp = requests.get(href, stream=True, timeout=30)
                                    resp.raise_for_status()
                                    with open(file_path, 'wb') as f:
                                        for chunk in resp.iter_content(chunk_size=8192):
                                            f.write(chunk)
                                    downloaded_files.append(file_path)
                                    print(f"[+] Saved (via requests): {file_path}")
                                except Exception as e2:
                                    print(f"[!] Requests fallback also failed for {filename}: {e2}")
                        else:
                            print(f"[~] Already exists: {file_path}")
                            downloaded_files.append(file_path)
            
            return downloaded_files

        except Exception as e:
            print(f"[!] Playwright error: {e}")
            return []
        finally:
            browser.close()

def download_via_requests(survey_ids, output_dir, max_files):
    """
    Uses direct HTTP requests to download BAG files from NCEI data servers.
    """
    downloaded_files = []
    
    # Base URL pattern for NOS BAG files
    # Example: https://data.ngdc.noaa.gov/platforms/ocean/nos/coast/H12001-H14000/H13607/BAG/H13607_MB_50cm_LWD_1of1.bag
    # We need to find the specific path for each survey ID.
    # The NCEI metadata API can provide the download URL.
    
    for survey_id in survey_ids[:max_files]:
        # Query metadata for this specific survey to get the BAG download URL
        meta_url = f"https://www.ncei.noaa.gov/access/metadata/portal/api/v1/survey/{survey_id}"
        try:
            resp = requests.get(meta_url, timeout=30)
            resp.raise_for_status()
            meta_data = resp.json()
            
            # Look for BAG file links in the metadata
            # The structure varies, but often there's a 'distribution' or 'download' section
            links = []
            if isinstance(meta_data, dict):
                # Check various possible keys for links
                for key in ['distribution', 'download', 'links', 'data']:
                    if key in meta_data:
                        val = meta_data[key]
                        if isinstance(val, list):
                            links.extend(val)
                        elif isinstance(val, dict):
                            links.append(val)
            
            bag_url = None
            for link in links:
                if isinstance(link, dict):
                    href = link.get('href') or link.get('url') or link.get('downloadUrl')
                    if href and '.bag' in href.lower():
                        bag_url = href
                        break
            
            if not bag_url:
                # Try to construct URL based on common patterns if metadata doesn't have direct link
                # This is heuristic and may break.
                print(f"[!] No direct BAG URL found for {survey_id} in metadata.")
                continue
                
            filename = os.path.basename(bag_url)
            file_path = os.path.join(output_dir, filename)
            
            if not os.path.exists(file_path):
                print(f"[*] Downloading {filename} from {bag_url}...")
                resp = requests.get(bag_url, stream=True, timeout=60)
                resp.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                downloaded_files.append(file_path)
                print(f"[+] Saved: {file_path}")
            else:
                print(f"[~] Already exists: {file_path}")
                downloaded_files.append(file_path)
                
        except Exception as e:
            print(f"[!] Error processing survey {survey_id}: {e}")
            
    return downloaded_files

def main():
    parser = argparse.ArgumentParser(description="Download BAG files from NCEI")
    parser.add_argument("--bbox", nargs=4, type=float, required=True,
                        help="Bounding box: lat_min lon_min lat_max lon_max")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for BAG files")
    parser.add_argument("--max-files", type=int, default=20,
                        help="Maximum number of files to download")
    
    args = parser.parse_args()
    
    lat_min, lon_min, lat_max, lon_max = args.bbox
    output_dir = args.output
    max_files = args.max_files
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[*] Bounding Box: ({lat_min}, {lon_min}) to ({lat_max}, {lon_max})")
    print(f"[*] Output Directory: {output_dir}")
    
    # Step 1: Try to find surveys via API
    print("[*] Querying NCEI API for surveys in bbox...")
    survey_ids = find_surveys_in_bbox(lat_min, lon_min, lat_max, lon_max)
    
    if not survey_ids:
        print("[!] No surveys found via API. Falling back to Playwright interaction.")
        # If API fails, we still try Playwright to see if it can find files
        # But without survey IDs, Playwright's ability to find specific files is limited.
        # We'll run Playwright anyway to see if it can surface any links.
        downloaded = download_via_playwright(args.bbox, output_dir, max_files)
        if downloaded:
            print(f"\n[+] Manifest of downloaded files:")
            for f in downloaded:
                print(f"  - {f}")
            return
        else:
            print("[!] Playwright also failed to find files. Exiting.")
            sys.exit(1)
    else:
        print(f"[+] Found {len(survey_ids)} potential surveys: {survey_ids[:5]}...")
        
        # Step 2: Download via direct requests (preferred)
        print("[*] Downloading BAG files via direct HTTP requests...")
        downloaded = download_via_requests(survey_ids, output_dir, max_files)
        
        if not downloaded:
            print("[!] Direct download failed or no files found. Trying Playwright as fallback...")
            downloaded = download_via_playwright(args.bbox, output_dir, max_files)
    
    print(f"\n[+] Manifest of downloaded files:")
    for f in downloaded:
        print(f"  - {f}")
    
    if not downloaded:
        print("[!] No files were downloaded.")
        sys.exit(1)

if __name__ == "__main__":
    main()
```
