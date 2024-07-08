import os
import zarr

# Define the datasets and the files to look for
dataset_names = ['MD17','MD17_vext_v0']
filenames = ['000002.zar.zip',
            '000013.zarr.zip',
            '000016.zarr.zip',
            '000020.zarr.zip',
            '000041.zarr.zip',
            '000042.zarr.zip',
            '000043.zarr.zip',
            '000051.zarr.zip',
            '000064.zarr.zip',
            '000065.zarr.zip']

filenames_v0 = ['000002.000001.zarr.zip',
            '000013.000001.zarr.zip',
            '000016.000001.zarr.zip',
            '000020.000001.zarr.zip',
            '000041.000001.zarr.zip',
            '000042.000001.zarr.zip',
            '000043.000001.zarr.zip',
            '000051.000001.zarr.zip',
            '000064.000001.zarr.zip',
            '000065.000001.zarr.zip']

# Define the base directory
base_dir = os.getenv('DFT_DATA')

for file_name, file_name_v0 in zip(filenames, filenames_v0):
    for dataset in dataset_names:
        labels_dir = os.path.join(base_dir, dataset, 'labels')
        # Determine the correct file name based on the dataset
        if dataset == 'MD17':
            file_path = os.path.join(labels_dir, file_name)
        elif dataset == 'MD17_vext_v0':
            file_path = os.path.join(labels_dir, file_name_v0)

        # Check if this file exists
        if os.path.exists(file_path):
            # Open the Zarr archive
            zarr_archive = zarr.open(file_path, mode='r')
            has_energy_label = zarr_archive['ks_labels']['energies']['has_energy_label'][:]
            e_tot = zarr_archive['of_labels']['energies']['e_tot'][:][has_energy_label]

            # Print energies directly within the loop
            print(f"Energies from {dataset}, file {file_name}:")
            print(e_tot)

            