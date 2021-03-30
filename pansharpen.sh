#!/bin/bash

# Variables
origin_dir=$1
output_dir=$2
search_term=$3

# Search files
list=$(find "$origin_dir" -maxdepth 2 -type d -name "*$search_term*")

mkdir "$output_dir"

for folder in $list
do
	ms_tif=$(find "$folder" -name "*ORT_MS*.TIF")
	p_tif=$(find "$folder" -name "*ORT_P*.TIF")

	echo "Panchromatic: ${p_tif[0]}"
	echo "Multispectral: ${ms_tif[0]}"

	basename_tif=$(basename "${p_tif[0]}")
	echo "Storing result in $output_dir/$basename_tif"
	echo " "

	gdal_pansharpen.py "${p_tif[0]}" "${ms_tif[0]}" "$output_dir/$basename_tif"
done
