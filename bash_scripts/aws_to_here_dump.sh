#!/bin/bash

targert_dir="/media/jagraz/KPMP_Data/Privately_Available_Data/Original_Download_KPMP_S3_Bucket_Oct_16_2023/requested_samples"
source_dir="s3://kpmp-knowledge-environment"

aws s3 cp $source_dir/e59bd96c-6d3b-40c4-8852-6c5b04112a94/61ebb98a-892b-4e49-a80f-66a68897a20e_S-1910-000142_KL-0015332.bam $targert_dir/"e59bd96c-6d3b-40c4-8852-6c5b04112a94/"
aws s3 cp $source_dir/659944f2-e186-48ef-b810-164a33c11320/6f83591d-5e1c-4c7a-b86b-198b1eda6701_S-2001-000048_KL-0016006.bam $targert_dir/"659944f2-e186-48ef-b810-164a33c11320/"
aws s3 cp $source_dir/4aeb404b-1026-43c9-9617-abb92354b6e6/3b1763d0-26dd-44cd-866b-028638ea99cb_S-1908-000945_KL-0014796.bam $targert_dir/"4aeb404b-1026-43c9-9617-abb92354b6e6/"
aws s3 cp $source_dir/3f6e320d-3864-4239-aef1-79dae21709a1/8ed512b0-b608-4958-8f39-1c85c013056a_S-1910-000189_KL-0015333.bam $targert_dir/"3f6e320d-3864-4239-aef1-79dae21709a1/"
aws s3 cp $source_dir/60725d76-ec9e-46f8-941c-4c5ec221d93b/87ef55ff-f5c8-4c62-b35a-b04a2eee516d_S-1910-000095_KL-0015331.bam $targert_dir/"60725d76-ec9e-46f8-941c-4c5ec221d93b/"
aws s3 cp $source_dir/cb72ee48-d705-49dd-a603-43cef7144be2/94488cea-9c22-4a6a-8c66-93f704bc8dc4_S-1904-008134_KL-0013915.bam $targert_dir/"cb72ee48-d705-49dd-a603-43cef7144be2/"
aws s3 cp $source_dir/ebda7973-2c50-435b-8b82-31910724300b/0fe76651-f62d-4f7b-bf20-27c7c6ad2bf8_S-1908-009883_KL-0014954.bam $targert_dir/"ebda7973-2c50-435b-8b82-31910724300b/"
aws s3 cp $source_dir/1062b763-01b1-4186-be03-af05f808a3de/620e848e-5bc4-4ccf-84e7-b63ef9e85772_S-1908-010071_KL-0014958.bam $targert_dir/"1062b763-01b1-4186-be03-af05f808a3de/"
aws s3 cp $source_dir/7b9a2765-3299-4ef2-8324-41ecb6808b35/efbac6f7-f1c1-4053-af8b-62e38d4f06bf_S-1908-009646_KL-0014947_possorted_genome_bam.bam $targert_dir/"7b9a2765-3299-4ef2-8324-41ecb6808b35/"
