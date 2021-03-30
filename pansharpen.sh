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
	foldername=$(basename "$folder")
	ms_tif=$(find "$folder" -name "*ORT_MS*.TIF")
	# shellcheck disable=SC2086
	p_tif=$(find $folder -name "*ORT_P*.TIF")

	# shellcheck disable=SC2086
	echo ${p_tif[0]}
	echo "${ms_tif[0]}"

	# shellcheck disable=SC2086
	gdal_pansharpen.py "${p_tif[0]}" ${ms_tif[0]} "$output_dir/$foldername.TIF"
done
