       program re_cal
       implicit real*8 (a-h,o-z)
       parameter(maxdata=50000)
       parameter(maxir=100000)
       real*8 fo(maxdata),fe(maxdata)
       real*8 wave_num(maxdata)
       real*8 sumdata(maxdata)
       real*8 wave(maxir),fir(maxir)
       character*30 in_file
       convfac=log(10.0)*1.0d0/(6.022*(10.0**20))
       k_count=0
       in_file='pinnock_table.txt'

       i_count=0
       open(1,file=in_file,status='old')
       do k=1,maxdata
          read(1,*,end=100)wave_num(k),fo(k),fe(k)
c       write(*,*)wave_num(k),fe(k)
          i_count=i_count+1
       enddo
100    continue
       do k=1,maxir
          read(*,*,end=200)wave(k),fir(k)
c          wave=wave*0.977+11.664
c          do i=1,500
c           wave_num=dfloat(i)*5.0d0
c           if((wave.gt.wave_num-5.0d0) .or.
c     &     (wave.lt.wave_num+5.0d0)) then
c            write(*,*)wave_num,fir
c           end if
c          enddo
           fir(k)=convfac*fir(k)
           write(9,*)wave(k),fir(k)*(10.0**18)
c           wave(k)=wave(k)*0.977+11.664
c           if(wave(k).ge.0.0d0)then
c           endif
c           if(wave(k).ge.0.0d0)then
           write(8,*)wave(k),fir(k)*(10.0**18)
           k_count=k_count+1
c           endif
       enddo
       dx=wave(k_count)-wave(k_count-1)
c       write(*,*)dx
200    continue
        sumout=0.0
       do i=1,i_count
        dsum=0.0d0
              
        aa=wave_num(i)
        dfe=fe(i)
        j_count=0
c        sumout=0.0
        do k=1,k_count
         aa=wave_num(i)
         bb=wave(k)
         cc=aa-5.0d0
         dd=aa+5.0d0
         dat=fir(k)
c         write(*,*)aa
         if((bb.ge.cc).and.(bb.le.dd)) then 
c         write(*,*)aa,dat*dfe
          j_count=j_count+1
          dsum=dsum+dat*dfe
c         write(*,*)aa,fir(k)*fe
         endif
        enddo
         ff=dfloat(j_count)
c        write(*,*)j_count
        if(ff.gt.0.0d0) then
        dsum=dsum/ff
        sumout=sumout+dsum
        write(*,*)aa,dsum,sumout
        endif
       enddo
       write(*,*)sumout*10.0d0*(1.0d15)
       stop
       end
