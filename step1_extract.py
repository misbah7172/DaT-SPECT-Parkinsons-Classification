"""Step 1: Extract NIfTIs from zip to D: drive."""
import os
import time
import zipfile
import shutil

ZIP_PATH = "E:/DaT/Dataset/DaT_Parkinsons_Challenge_-_niftis.zip"
EXTRACT_DIR = "D:/DaT_cache/niftis"

def main():
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    
    # Check if already extracted
    existing = [f for f in os.listdir(EXTRACT_DIR) if f.endswith('.nii.gz')]
    if len(existing) > 1000:
        print(f"Already extracted {len(existing)} files")
        return
    
    print(f"Extracting NIfTIs to {EXTRACT_DIR}...")
    t0 = time.time()
    
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            names = zf.namelist()
            print(f"Files in zip: {len(names)}")
            
            for i, name in enumerate(names, 1):
                if name.endswith('.nii.gz'):
                    # Extract to flat structure
                    basename = os.path.basename(name)
                    out_path = os.path.join(EXTRACT_DIR, basename)
                    if not os.path.exists(out_path):
                        with zf.open(name) as src, open(out_path, 'wb') as dst:
                            dst.write(src.read())
                
                if i % 100 == 0:
                    elapsed = time.time() - t0
                    rate = i / elapsed
                    eta = (len(names) - i) / rate
                    print(f"  [{i}/{len(names)}] {elapsed:.0f}s elapsed, ETA={eta:.0f}s")
        
        total = time.time() - t0
        final_count = len([f for f in os.listdir(EXTRACT_DIR) if f.endswith('.nii.gz')])
        print(f"Done: {final_count} NIfTI files extracted in {total:.0f}s")
    except PermissionError:
        print("Permission denied on zip file. Trying alternative...")
        # Try copying the zip file first
        alt_zip = "D:/DaT_cache/dat_niftis.zip"
        print(f"Copying zip to {alt_zip}...")
        shutil.copy2(ZIP_PATH, alt_zip)
        print("Extracting from copy...")
        with zipfile.ZipFile(alt_zip, "r") as zf:
            zf.extractall(EXTRACT_DIR)
        print("Done extracting from copy")

if __name__ == "__main__":
    main()
