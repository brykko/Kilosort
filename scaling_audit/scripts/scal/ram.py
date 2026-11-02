import math
def C(n): return 1e6*2**max(0,math.ceil(math.log2(n/1e6)))
G=2**30
print('T     Nu    Nl   C(Nu) C(Nl) | A drift  B univ  C learnDbl D gather E merge  F1 pos  F2 pcfeat  (GB, live arrays only)')
for f,lab in [(1,'20m'),(3,'1h'),(6,'2h'),(7.2,'2.4h'),(9,'3h'),(15,'5h')]:
    Nu=4.02e6*f; Nl=8.74e6*f; Cu=C(Nu); Cl=C(Nl)
    A=2*Cu*288
    B=2*Cu*288+Cu*48
    Cc=2*Cl*264+Cu*288+Cu*48+Cu*48
    D=Cl*264+Nl*264+Cu*288+Cu*48+Cu*48
    E=2*Nl*264+Cu*48+Nu*4
    F1=Nl*(240+24+260)+Cu*48
    F2=Nl*(3*240+40)+Cu*48
    print(f'{lab:5} {Nu/1e6:5.1f} {Nl/1e6:6.1f} {Cu/1e6:4.0f} {Cl/1e6:4.0f} | ' + '  '.join(f'{x/G:6.1f}' for x in (A,B,Cc,D,E,F1,F2)))
