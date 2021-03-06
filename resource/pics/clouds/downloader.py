#!/usr/bin/python3.4

import urllib
from datetime import date, timedelta

'''
	urllib.request.urlretrieve

	http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/2014/10/S11235251_201410160000.jpg

'''

patch_template = ['http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}2330.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}2300.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}2230.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}2200.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}2130.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}2100.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}2030.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}2000.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}1930.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}1900.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}1830.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}1800.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}1730.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}1630.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}1600.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}1500.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}1430.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}1400.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}1330.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}1300.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}1230.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}1200.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}1130.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}1100.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}1030.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}1000.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}0930.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}0900.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}0830.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}0800.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}0730.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}0700.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}0630.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}0600.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}0530.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}0500.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}0400.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}0330.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}0300.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}0230.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}0200.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}0130.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}0100.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}0030.jpg',
'http://satelite.cptec.inpe.br/repositorio5/goes13/goes13_web/ne_infra_alta/{ano}/{mes}/S11235251_{ano}{mes}{dia}0000.jpg']

def main():
    ano, mes, dia = 2013, 4, 1
    moment = date(ano, mes, dia)

    while(moment.month < 5):
        for h in patch_template:
            url = h.format(ano=moment.strftime("%Y"), mes=moment.strftime("%m"), dia=moment.strftime("%d"))

            try:
                urllib.request.urlretrieve(url, url.split('/')[9])
                print(url.split('/')[9] + " -- " + moment.strftime("%d %m %Y"))
            except Exception:
                print("404 HTTP " + url.split('/')[9])
        #Soma um dia ao valor da data
        moment = moment + timedelta(1)

if __name__ == '__main__':
	main()
