"""
Comprehensive DICOM Series Audit and Exploration Script for ADNI MPRAGE Benchmark.
Inspects all 59 subjects and all DICOM files under D:\\AIML\\ADNI_MPRAGE_Benchmark_Extracted\\ADNI.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pydicom
from collections import defaultdict


def parse_dicom_series(series_dir, sub_id):
    files = [os.path.join(series_dir, f) for f in os.listdir(series_dir) if os.path.isfile(os.path.join(series_dir, f))]
    if not files:
        return None

    # Sort files by filename as initial pass
    files.sort()
    
    # Read headers
    slice_data = []
    read_errors = 0

    first_dcm = None
    for fpath in files:
        try:
            # stop_before_pixels=True for fast header parsing
            dcm = pydicom.dcmread(fpath, stop_before_pixels=True)
            if first_dcm is None:
                first_dcm = dcm
            
            inst_num = getattr(dcm, "InstanceNumber", None)
            acq_num = getattr(dcm, "AcquisitionNumber", None)
            ipp = getattr(dcm, "ImagePositionPatient", None)
            iop = getattr(dcm, "ImageOrientationPatient", None)
            
            slice_data.append({
                "filepath": fpath,
                "filename": os.path.basename(fpath),
                "instance_number": int(inst_num) if inst_num is not None else None,
                "acquisition_number": int(acq_num) if acq_num is not None else None,
                "image_position_patient": [float(x) for x in ipp] if ipp is not None else None,
                "image_orientation_patient": [float(x) for x in iop] if iop is not None else None,
            })
        except Exception as e:
            read_errors += 1

    if not slice_data or first_dcm is None:
        return None

    # Series-level metadata from first slice
    series_uid = getattr(first_dcm, "SeriesInstanceUID", "")
    series_desc = getattr(first_dcm, "SeriesDescription", "")
    protocol_name = getattr(first_dcm, "ProtocolName", "")
    image_type = getattr(first_dcm, "ImageType", "")
    if isinstance(image_type, (list, tuple, pydicom.multival.MultiValue)):
        image_type_str = r"\ ".join([str(x) for x in image_type])
    else:
        image_type_str = str(image_type)
        
    acq_num = getattr(first_dcm, "AcquisitionNumber", None)
    modality = getattr(first_dcm, "Modality", "")
    manufacturer = getattr(first_dcm, "Manufacturer", "")
    field_strength = getattr(first_dcm, "MagneticFieldStrength", None)
    if field_strength is not None:
        try:
            field_strength = float(field_strength)
        except:
            pass
            
    rows = getattr(first_dcm, "Rows", None)
    cols = getattr(first_dcm, "Columns", None)
    slice_thick = getattr(first_dcm, "SliceThickness", None)
    if slice_thick is not None:
        try:
            slice_thick = float(slice_thick)
        except:
            pass
            
    pixel_spacing = getattr(first_dcm, "PixelSpacing", None)
    if pixel_spacing is not None:
        pixel_spacing_list = [float(x) for x in pixel_spacing]
        pixel_spacing_str = f"{pixel_spacing_list[0]:.4f}\\{pixel_spacing_list[1]:.4f}"
    else:
        pixel_spacing_list = None
        pixel_spacing_str = ""

    spacing_between_slices = getattr(first_dcm, "SpacingBetweenSlices", None)
    if spacing_between_slices is not None:
        try:
            spacing_between_slices = float(spacing_between_slices)
        except:
            pass

    study_date = getattr(first_dcm, "StudyDate", "")
    series_num = getattr(first_dcm, "SeriesNumber", None)
    if series_num is not None:
        try:
            series_num = int(series_num)
        except:
            pass

    iop = getattr(first_dcm, "ImageOrientationPatient", None)
    if iop is not None:
        iop_list = [round(float(x), 4) for x in iop]
    else:
        iop_list = None

    # Assess slice spatial continuity and completeness
    has_all_ipp = all(s["image_position_patient"] is not None for s in slice_data)
    has_all_inst = all(s["instance_number"] is not None for s in slice_data)
    
    inst_nums = [s["instance_number"] for s in slice_data if s["instance_number"] is not None]
    duplicate_instances = len(inst_nums) != len(set(inst_nums))
    
    # Check if instance numbers form an unbroken sequence
    if has_all_inst and len(inst_nums) > 0:
        min_inst = min(inst_nums)
        max_inst = max(inst_nums)
        expected_count = max_inst - min_inst + 1
        inst_continuous = (expected_count == len(inst_nums)) and not duplicate_instances
    else:
        min_inst, max_inst = None, None
        inst_continuous = False

    # Check slice position spacing along normal
    z_spacings = []
    is_spatial_continuous = False
    sorted_positions = []
    if has_all_ipp and iop_list is not None and len(iop_list) == 6:
        # Normal vector is cross product of row and col vectors
        r_vec = np.array(iop_list[:3])
        c_vec = np.array(iop_list[3:])
        normal_vec = np.cross(r_vec, c_vec)
        norm = np.linalg.norm(normal_vec)
        if norm > 1e-6:
            normal_vec = normal_vec / norm
            
            # Project IPP onto normal vector
            for s in slice_data:
                pos = np.array(s["image_position_patient"])
                proj = np.dot(pos, normal_vec)
                sorted_positions.append(proj)
            sorted_positions.sort()
            
            diffs = np.diff(sorted_positions)
            z_spacings = [float(d) for d in diffs]
            if len(diffs) > 0:
                mean_spacing = float(np.mean(diffs))
                std_spacing = float(np.std(diffs))
                # If std is small, spacing is regular
                is_spatial_continuous = (std_spacing < 0.05) and (len(slice_data) >= 100)
            else:
                mean_spacing = 0.0
                std_spacing = 0.0
        else:
            mean_spacing = None
            std_spacing = None
    else:
        mean_spacing = None
        std_spacing = None

    # Check for orientation consistency across all slices
    unique_iops = set(tuple(s["image_orientation_patient"]) for s in slice_data if s["image_orientation_patient"] is not None)
    orientation_consistent = (len(unique_iops) == 1)

    # Completeness check
    is_complete = (
        (read_errors == 0) and
        has_all_ipp and
        has_all_inst and
        not duplicate_instances and
        inst_continuous and
        is_spatial_continuous and
        orientation_consistent and
        len(slice_data) >= 120 # typical 3D volume has >= 120 slices
    )

    series_id = os.path.basename(series_dir) # e.g. I25405

    return {
        "subject_id": sub_id,
        "series_id": series_id,
        "series_path": series_dir,
        "series_instance_uid": str(series_uid),
        "series_description": str(series_desc),
        "protocol_name": str(protocol_name),
        "image_type": image_type_str,
        "acquisition_number": int(acq_num) if acq_num is not None else None,
        "modality": str(modality),
        "manufacturer": str(manufacturer),
        "field_strength": field_strength,
        "rows": int(rows) if rows is not None else None,
        "columns": int(cols) if cols is not None else None,
        "slice_thickness": slice_thick,
        "pixel_spacing": pixel_spacing_str,
        "spacing_between_slices": spacing_between_slices,
        "calculated_slice_spacing": round(mean_spacing, 4) if mean_spacing is not None else None,
        "slice_spacing_std": round(std_spacing, 6) if std_spacing is not None else None,
        "num_slices": len(slice_data),
        "read_errors": read_errors,
        "study_date": str(study_date),
        "series_number": series_num,
        "image_orientation_patient": str(iop_list) if iop_list is not None else "",
        "instance_min": min_inst,
        "instance_max": max_inst,
        "inst_continuous": inst_continuous,
        "spatial_continuous": is_spatial_continuous,
        "orientation_consistent": orientation_consistent,
        "is_complete": is_complete,
    }


def main():
    root = r"D:\AIML\ADNI_MPRAGE_Benchmark_Extracted\ADNI"
    subjects = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    print(f"Auditing {len(subjects)} subjects under {root}...")

    all_series = []
    sub_series_map = defaultdict(list)

    for sub in subjects:
        sub_dir = os.path.join(root, sub)
        # Find all series directories (directories containing dicom files)
        for dirpath, dirnames, filenames in os.walk(sub_dir):
            if filenames and any(f.endswith(".dcm") or not os.path.splitext(f)[1] for f in filenames):
                info = parse_dicom_series(dirpath, sub)
                if info:
                    all_series.append(info)
                    sub_series_map[sub].append(info)

    print(f"Audit completed: Found {len(all_series)} series across {len(subjects)} subjects.")
    
    # Save raw audit DataFrame
    audit_df = pd.DataFrame(all_series)
    os.makedirs("results", exist_ok=True)
    audit_csv_path = os.path.join("results", "adni_series_audit.csv")
    audit_df.to_csv(audit_csv_path, index=False)
    print(f"Saved series audit to: {audit_csv_path}")

    # Inspect subjects with >1 series
    multi_series_subs = {s: s_list for s, s_list in sub_series_map.items() if len(s_list) > 1}
    print(f"\nSubjects with exactly 1 series: {len(subjects) - len(multi_series_subs)}")
    print(f"Subjects with >1 series: {len(multi_series_subs)}")
    
    for s, s_list in multi_series_subs.items():
        print(f"\nSubject {s} has {len(s_list)} series:")
        for idx, s_info in enumerate(s_list):
            print(f"  [{idx+1}] ID: {s_info['series_id']}, Desc: '{s_info['series_description']}', "
                  f"Protocol: '{s_info['protocol_name']}', Slices: {s_info['num_slices']}, "
                  f"Type: '{s_info['image_type']}', Acq#: {s_info['acquisition_number']}, "
                  f"Series#: {s_info['series_number']}, Thick: {s_info['slice_thickness']}, "
                  f"Spacing: {s_info['calculated_slice_spacing']}, Date: {s_info['study_date']}, "
                  f"Complete: {s_info['is_complete']}")


if __name__ == "__main__":
    main()
