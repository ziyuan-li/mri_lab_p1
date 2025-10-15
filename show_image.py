import os
import pydicom
import matplotlib.pyplot as plt

# dicom_dir = r"C:\Li\Projects\tum_courses\mri_lab_p1\data\MRI LAB G2 water\1.2.826.0.1.3680043.8.1276.259245911.266.1000\1.2.826.0.1.3680043.8.1276.259245911.266.1000.6\Dicoms"
# dicom_dir = r"C:\Li\Projects\tum_courses\mri_lab_p1\data\Water Group 2 second try\1.2.826.0.1.3680043.8.1276.2592454444.267.1000\1.2.826.0.1.3680043.8.1276.2592454444.267.1000.2\Dicoms"
dicom_dir = r"C:\Li\Projects\tum_courses\mri_lab_p1\data\Group 2 Gadulinium T1\1.2.826.0.1.3680043.8.1276.259246159.268.1000\1.2.826.0.1.3680043.8.1276.259246159.268.1000.10\Dicoms"

dicom_files = sorted([f for f in os.listdir(dicom_dir) if f.endswith(".dcm")])

for file_name in dicom_files:
    file_path = os.path.join(dicom_dir, file_name)
    ds = pydicom.dcmread(file_path)

    print(f"=== {file_name} ===")
    def g(tag, default="N/A"):
        return ds.get(tag, default)

    print("Patient ID:", g("PatientID"))
    print("Study Date:", g("StudyDate"))
    print("Modality:", g("Modality"))
    print("Manufacturer:", g("Manufacturer"))
    print("Series Description:", g("SeriesDescription"))
    print("Sequence Name:", g("SequenceName"))
    print("Repetition Time (TR) [ms]:", g("RepetitionTime"))
    print("Echo Time (TE) [ms]:", g("EchoTime"))
    print("Inversion Time (TI) [ms]:", g("InversionTime"))
    print("Flip Angle [deg]:", g("FlipAngle"))
    print("Slice Thickness [mm]:", g("SliceThickness"))
    print("Pixel Spacing [mm]:", g("PixelSpacing"))
    print("Image Size (Rows x Cols):", ds.Rows, "x", ds.Columns)
    print("\n\n")

    plt.imshow(ds.pixel_array, cmap="gray")
    plt.axis("off")
    plt.show()
