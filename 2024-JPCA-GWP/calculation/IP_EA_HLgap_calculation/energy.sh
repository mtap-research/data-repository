#!/bin/bash


for i in {0..199} 
do
 grep B3LYP neutral_$i.log |head -n 1 >aaa
 awk '{print $5}' aaa >>neutral_en.out
 grep B3LYP cation_$i.log |head -n 1 >aaa
 awk '{print $5}' aaa >>cation_en.out
 grep B3LYP anion_$i.log |head -n 1 >aaa
 awk '{print $5}' aaa >>anion_en.out
done


