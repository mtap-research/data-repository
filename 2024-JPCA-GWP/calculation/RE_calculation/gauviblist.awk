# AWK Program
# 
# This Program reads a G92 output file and lists vibrations and intesities
# in a form suitable for using them with irspec.x
#
# Author: Stephan Matzinger, Mar 1995
#

# search for "Normal mode"


/Harmonic frequencies/  { normmo+=1 ; next }

normmo == 1 && /Frequencies/ { 
	anzahl=NF-2
	altvib=totalvib
        totalvib+=anzahl 
        for(i=1; i<=anzahl; i++)
		{
		j=altvib+i
		k=i+2 
		freq[j]=$k
		}
        }

normmo == 1 && /IR Inten/ { 
        for(i=1; i<=anzahl; i++)
                {
                j=altvib+i
                k=i+3
                intens[j]=$k
                }
	}

END	{
	printf("%d \n",totalvib)
	printf("1.00\n1.00\n14.0\n")
	for(i=totalvib; i>=1; --i) 	
		printf("%8.2f%10.4f\n",freq[i],intens[i])
	}
