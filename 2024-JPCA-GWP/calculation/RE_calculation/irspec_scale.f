        program calcir
        implicit none
	real*8  intens(5000),omega,pi
	real*8 position(5000)
	real*8  spectrum(4000) ,wave(4000)
	real*8  scalefreq,scaleint
	integer nvib
	integer i,j,nn,wavemax
        character jobname*50


        call getarg (1,jobname)

        nn=0
    5   nn = nn + 1
        IF (jobname(nn:nn).NE.' ') GOTO 5
        nn = nn - 1

        open (1,file=jobname(1:nn)//'.ir',
     &    form='FORMATTED',status='old')


        pi=dacos(-1.0d0)
	do i=1,4000 
	   spectrum(i)=0.0
           wave(i)    =0.0
        end do


C How many vibrations? '',$)')
	read (1,'(I4)') nvib
C Scale factor for Frequency? '',$)')
	read (1,'(F10.5)') scalefreq
C Scale factor for Intensity? '',$)')
	read (1,'(F10.5)') scaleint
C Linewidth omega?
        read (1,'(F10.5)') omega

        do i=1,nvib
            read (1,'(F8.2,F10.4)') position(i),intens(i)
            write (9,'(F8.2,F10.5)') position(i),intens(i)
            position(i)=position(i)*scalefreq
        end do


C   ********** Input done

C  *** Code for Lorentzian **** (omega .GT. 0)
       if (omega .GT. 0) then

        do j=1,4000
           do i=1,nvib
              wave(j) = j
c              j=0.977*j+11.664
              write(*,*) position(i)
              if(position(i).ge.0.0d0) then
              spectrum(j)=spectrum(j)+scaleint*intens(i)*
     &            (42.2561*2.0/pi)*
     &            omega/(4*(j-(position(i)*0.977+11.664))**2+omega**2)
      endif
           end do
        end do
       wavemax = 4000


C *** Code for Line Spectrum *** (omega .EQ. 0)
      else
        wave(1) = 0.0
        j=3
        do i=1,nvib
            wave(j-1) = position (i) - 0.01
            wave(j)   = position (i)
            spectrum(j)=scaleint*intens(i)
            wave(j+1) = position (i) + 0.01
            j=j+3
        end do
        wave(j-1) = 4000.0
        wavemax = j-1
      end if



C  **** Output directly to file

        open (2,file=jobname(1:nn)//'.prn',
     &    form='FORMATTED',status='unknown')

        do j=1,wavemax
          write(2,'(F8.2,F14.5)')  wave(j),spectrum(j)
c          write(*,'(F8.2,F14.5)')  wave(j),spectrum(j)
        end do

        close(1)
c        close(2)


	end


