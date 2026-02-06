import os
import pandas as pd
import nibabel as nib
import numpy as np
import dicom2nifti
from nilearn import image, datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil


RAW_MRI_DIR = "MRI"
METADATA_CSV = "mri_metadata.csv"
OUTPUT_DIR = "processed_dataset"
TARGET_SHAPE = (128, 128, 128)

def run_pipeline():
 
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

 
    df = pd.read_csv(METADATA_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.drop_duplicates(subset=['subject'], keep='first')
    label_map = {str(k).strip(): str(v).strip() for k, v in zip(df['subject'], df['group'])}

   
    data_list = []
    for root, dirs, files in os.walk(RAW_MRI_DIR):
        if files:
            for sid in label_map.keys():
                if sid in root:
                    data_list.append({'id': sid, 'path': root, 'label': label_map[sid]})
                    break
    
   
    data_list = list({v['id']: v for v in data_list}.values())
    print(f"📈 Found {len(data_list)} subjects. Starting processing...")

  
    mni_template = datasets.load_mni152_template()
    gm_mask = datasets.load_mni152_gm_mask()

    for item in tqdm(data_list):
        try:
        
            dest = os.path.join(OUTPUT_DIR, item['label'])
            os.makedirs(dest, exist_ok=True)

            temp_nii = f"temp_{item['id']}.nii.gz"
            dicom2nifti.dicom_series_to_nifti(item['path'], temp_nii, reorient_nifti=True)

          
            resampled = image.resample_to_img(temp_nii, mni_template)
            gm_img = image.math_img("img * mask", img=resampled, mask=gm_mask)

           
            img_data = gm_img.get_fdata()
            c_x, c_y, c_z = np.array(img_data.shape) // 2
            t_x, t_y, t_z = np.array(TARGET_SHAPE) // 2
            cropped = img_data[c_x-t_x:c_x+t_x, c_y-t_y:c_y+t_y, c_z-t_z:c_z+t_z]

            
            denom = (cropped.max() - cropped.min()) + 1e-8
            processed = (cropped - cropped.min()) / denom

           
            final_name = f"{item['id']}_clean.nii.gz"
            nib.save(nib.Nifti1Image(processed, mni_template.affine), os.path.join(dest, final_name))
            
         
            if os.path.exists(temp_nii): os.remove(temp_nii)

        except Exception as e:
            print(f"Skipping {item['id']} due to error: {e}")

    print(f" Pipeline Complete! Data is in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    run_pipeline()
