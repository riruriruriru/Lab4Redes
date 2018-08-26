import numpy as np
from numpy import sin, linspace, pi, cumsum, random
from scipy.io.wavfile import read, write
from scipy.io import wavfile
from scipy import fft, arange, ifft
from scipy import signal
from scipy.fftpack import fftshift
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy.interpolate import interp1d
import scipy.integrate as integrate

import matplotlib.pyplot as plt
def menu():
	#Muestra el menu principal y usando input se pide al usuario que ingrese la opcion que desea ejecutar
	option = 0
	while option == 0:
		print('Menu Principal')
		print('Opciones:')
		print('1) Ingresar nombre de archivo de audio')
		print('2) Salir')
		user_input = input('Ingrese el numero de la opcion que desea ejecutar: ')
		if user_input=="2":
			#se retorna y finaliza el programa
			return 0
		elif user_input=="1":
			#se pide ingresar un nombre de archivo
			error = 1
			while error == 1:
				input_nombre = input('Ingrese nombre del archivo de audio: ')
                #la apertura se realiza inicialmente en un try-except, para evitar que el programa se caiga
				#en caso de que el archivo no pueda ser abierto
				try:
					open(input_nombre, 'rb')
					error = 0
				except FileNotFoundError:
					error = 1       
					print('nombre de archivo no existente o fuera de directorio')
			#llamado a funcion que abre un archivo verificado y retorna datos de este
			rate, info, data , timp, t = process_audio(input_nombre)
			binArray = decToBinary(data)
			#indice = int(np.ceil((10**4)/len(binArray[0])))
			#print("------_////////////-------------")
			#print("indice")
			#print(indice)
			#print("------_//////----------")
			#binArray = binArray[:indice]
			binFlat = transform_to_int(binArray)
			tiempo = np.arange(0.001/100,(0.001)*10000+0.001/100,0.001/100)
			option = second_menu(rate, info, data, timp, t, binFlat[:10000], tiempo[:10000])

		else:
			print('Ingrese una opcion correcta')
            
	return
            
def second_menu(rate, info, data, timp, t, binFlat, tiempo):
	option = 0
	#Segundo menu que se llama cuando se abre correctamente un archivo
	while option == 0:
		print('##########################')
		print('Menu Archivo')
		print('Opciones:')
		print('1) Mostrar Gráfico Binarizado')
		print('2) Modulacion ASK digital')
		print('3) Retroceder al menu anterior')
		print('4) Salir')
		user_input = input('Ingrese el numero de la opcion que desea ejecutar: ')
		if user_input=="3":
			#opcion que sirve para retroceder al menu anterior y abrir un nuevo archivo
			return 0
		elif user_input=="1":
			graficar_binario(binFlat,10, "Original")
		elif user_input=="2":
			option = third_menu_ask(rate, data, t, info, binFlat, tiempo)
		elif user_input=="4":
			return 2
		else:
			print("Ingrese una opcion correcta")
	return

def third_menu_ask(rate, data,  t, info, binFlat, tiempo):
	option = 0
	while option == 0:
		print('####################')
		print('Menu ASK')
		print('Opciones:')
		print('1) Aplicar modulacion ASK')
		print('2) Retroceder al menu anterior')
		print('3) Salir')
		m, largo, f, tiempoMod, bp = ask_modulation(binFlat[:10000], 10)
		user_input = input('Ingrese el numero de la opcion que desea ejecutar: ')
		if user_input=="1":
			graph(tiempoMod, m, "Tiempo[s]", "Bits", "Modulacion ASK")
			ASK_demodulation_menu(tiempoMod, m,bp,largo,f, data, rate, info)
		elif user_input=="2":
			return 0
		elif user_input=="3":
			return 2
		else:
			print('Ingrese una opcion valida')
	return

def ASK_demodulation_menu(tiempo, modulatedSignal, bp, largo, f, data, rate, info):
	option = 0
	while option == 0:
		print('####################')
		print('Menu ASK Demodulacion')
		print('Opciones:')
		print('1) Aplicar demodulacion ASK normal')
		print('2) Aplicar demodulacion ASK con ruido, SNR = 1')
		print('3) Aplicar demodulacion ASK con ruido, SNR = 2')
		print('4) Aplicar demodulacion ASK con ruido, SNR = 5')
		print('5) Aplicar demodulacion ASK con ruido, SNR = 8')
		print('6) Aplicar demodulacion ASK con ruido, SNR = 10')
		print('7) Retroceder al menu anterior')
		print('8) Salir')
		demod = ask_demodulation(bp, modulatedSignal, largo)
		user_input = input('Ingrese el numero de la opcion que desea ejecutar: ')
		if user_input=="1":
			graficar_binario(demod[:10000], bp, "Demodulada")
		elif user_input=="2":
			noisy = modulatedSignal+ awgn(len(modulatedSignal),1.0)
			demod2 = ask_demodulation(bp, noisy, largo)
			p=compare(demod, demod2)
			graphCompare(tiempo[:10000],demod[:10000],tiempo[:10000],demod2[:10000],"Tiempo[s]","Bits","Demodulacion sin ruido vs con ruido (snr = 1.0)", p, bp,1.0)
			
		
		elif user_input=="3":
			noisy = modulatedSignal+ awgn(len(modulatedSignal),2.0)
			demod2 = ask_demodulation(bp, noisy, largo)
			p=compare(demod, demod2)
			graphCompare(tiempo[:10000],demod[:10000],tiempo[:10000],demod2[:10000],"Tiempo[s]","Bits","Demodulacion sin ruido vs con ruido (snr = 2.0)", p, bp,2.0)
		
		elif user_input=="4":
			noisy = modulatedSignal+ awgn(len(modulatedSignal),5.0)
			demod2 = ask_demodulation(bp, noisy, largo)
			p=compare(demod, demod2)
			graphCompare(tiempo[:10000],demod[:10000],tiempo[:10000],demod2[:10000],"Tiempo[s]","Bits","Demodulacion sin ruido vs con ruido (snr = 5.0)", p, bp,5.0)
		
		elif user_input=="5":
			noisy = modulatedSignal+ awgn(len(modulatedSignal),8.0)
			demod2 = ask_demodulation(bp, noisy, largo)
			p=compare(demod, demod2)
			graphCompare(tiempo[:10000],demod[:10000],tiempo[:10000],demod2[:10000],"Tiempo[s]","Bits","Demodulacion sin ruido vs con ruido (snr = 8.0)", p, bp,8.0)
		
		elif user_input=="6":
			noisy = modulatedSignal+ awgn(len(modulatedSignal),10.0)
			demod2 = ask_demodulation(bp, noisy, largo)
			p=compare(demod, demod2)
			graphCompare(tiempo[:10000],demod[:10000],tiempo[:10000],demod2[:10000],"Tiempo[s]","Bits","Demodulacion sin ruido vs con ruido (snr = 10.0)", p, bp,10.0)
        
		elif user_input=="7":
			return 0                        
		elif user_input=="8":
			return 2
		else:
			print('Ingrese una opcion valida')
	return
#funcion que se encarga de recibir un audio y obtener sus datos
def process_audio(archivo):
    
    rate,info=wavfile.read(archivo)
    print('rate')
    print(rate)
    print(info)
    dimension = info[0].size
    print(dimension)
    if dimension==1:
        data = info
        perfect = 1
    else:
        data = info[:,dimension-1]
        perfect = 0
    timp = len(data)/rate
    t=linspace(0,timp,len(data))
    print("LARGO t")
    print(len(t))    
    return rate, info, data, timp, t
#funcion que recibe dos arreglos de la misma dimension y los grafica con las etiquetas tambien recibidas por argumento
def graph(x, y, labelx, labely, title):
	if title == "Modulacion ASK":
		plt.ylim(-1.5,1.5)
		plt.xlim(0, 5000)
	plt.title(title)
	plt.xlabel(labelx)
	plt.ylabel(labely)
	plt.plot(x, y)
	plt.show()
	return
def graphCompare(x, y,x2,y2, labelx, labely, title, percent, bp, snr):
    #plt.title(title)
    #plt.xlabel(labelx)
    #plt.ylabel(labely)
    #plt.plot(x, y)
    #plt.plot(x2,y2)
    #plt.show()
	yArray = []
	yArray2 = []
	for i in range(0, 100):
		for j in range(0,100):
			yArray.append(y[i])
	for i in range(0, 100):
		for j in range(0,100):
			yArray2.append(y2[i])
	t = np.arange(bp/100, bp*len(yArray) + bp/100, bp/100)
	t2 = np.arange(bp/100, bp*len(yArray2) + bp/100, bp/100)
	fig = plt.figure(1)
	fig.suptitle(title + " " +str(percent)+"%", fontsize=16)                # the first figure
	ax = plt.subplot(211)             # the first subplot in the first figure
	ax.set_title("Demodulada sin ruido")
	plt.plot(x[:10000],yArray[:10000])
	ax = plt.subplot(212)             # the second subplot in the first figure
	plt.plot(x2[:10000],yArray2[:10000])


	           # a second figure
	#plt.plot([4, 5, 6])          # creates a subplot(111) by default

	#plt.figure(1)                # figure 1 current; subplot(212) still current
	#plt.subplot(211)             # make subplot(211) in figure1 current
	
	plt.show()
	return

def fourier(rate, info, data):
     #funcion que transforma los datos obtenidos del archivo de sonidos al dominio de las frecuencias usando la transformada de fourier (rfft)
    timp = len(data)/rate
    large = len(data)
    #utilizando timp, se obtiene el tiempo total del archivo
    fourierTransform = np.fft.fftshift(fft(data))
    k = linspace(-len(fourierTransform)/2, len(fourierTransform)/2, len(fourierTransform))
    frq = k/timp
    print('largo de frecuencias ' + str(len(frq))+'largo datos ' + str(len(data)))
    #con k se obtiene un arreglo con tamanio igual al largo de los datos obtenidos con la transformada y se dividen por el tiempo, obteniendo un arreglo de frecuencias
    print(fourierTransform)   
    #se retorna el arreglo de frecuencias y datos de la transformada de fourier
    
    return frq, fourierTransform
    
#Entradas: datos correspondientes a la señal original
#Funcionamiento: Realiza una modulacion FM y grafica los resultados en el dominio del tiempo y de las frecuencias
#Salida nada
def FM_analog_modulation(rate, data, beta, t, info):
	#funcion que modula una señal con modulacion analoga FM
	title = str(beta*100) +"%"
	#crea un nuevo set de datos llamando a interpolate()
	data2 = interpolate(t, data, rate, info)
	newLen = len(data2)
	#crea un nuevo arreglo tiempo que tenga la misma dimension que el arreglo de datos
	newTime = np.linspace(0,len(data)/(rate), newLen)
	#frecuencia moduladora, debe ser mayor a la frecuencia original de muestreo
	fc = 30000
	#se define la portadora de la siguiente forma:
	carrier = np.sin(2*np.pi*newTime)
	w = fc*10*newTime
	#se integra
	integral = integrate.cumtrapz(data2, newTime, initial=0)
	#se obtiene la señal modulada
	resultado = np.cos(np.pi*w + beta*integral*np.pi)
	#grafico normal
	graph(t, data, "Tiempo[s]", "Amplitud[db]", "Tiempo vs Amplitud")
	#grafico modulado
	graph(newTime[1000:4000], resultado[1000:4000], "Tiempo[s]", "Amplitud[db]", "Modulacion FM al " +title)
	#grafico fourier
	freq, fourierT = fourier(rate*10, info, data)
	graph(freq, fourierT, "Frecuencia[hz]", "Magnitud de Frecuencia[db]", "Transformada de Fourier datos originales")
	#grafico fourier modulado
	#se grafica Frecuencia vs Magnitud
	freq2, fourierT2 = fourier(rate*10, info, resultado)
	graph(freq2, fourierT2, "tiempo", "frecuencia", "Transformada de fourier modulacion FM al " +title)
	return
#Entradas: datos correspondiente a la señal original
#Funcionamiento: se crea un nuevo arreglo de tiempo que refleja el muestreo que se quiere lograr, el cual se utiliza
#para obtener un nuevo arreglo de datos resampleado e interpolado
#Salidas: arreglo de datos resampleado
def interpolate(t, data, rate, info):
    interp = interp1d(t,data)
    newTime = np.linspace(0,len(data)/rate,len(data)*10)
    resultado = interp(newTime)
    return resultado
	
#Entradas: datos correspondientes a la señal original
#Funcionamiento: se realiza una modulacion AM analoga, mediante una interpolacion se remuestrea la señal y se grafican los resultados de la modulacion 
#en el dominio del tiempo y las frecuencias
#Salidas: datos resampleados y modulados
def AM_analog_modulation(rate,data,beta,t, info):
    title = str(beta*100) +"%"
    print("RATE: ")
    print(rate)
    #se interpola para generar un nuevo arreglo de datos
    data2 = interpolate(t, data, rate, info)
    fc=30000
    newLen = len(data2)
    #segun el largo del nuevo arreglo de datos, se obtiene un nuevo arreglo de tiempo
    newTime = np.linspace(0,len(data)/(rate), newLen)
    #se calcula la portadora
    carrier = np.cos(2*np.pi*newTime*fc)*beta
    #se multiplica la portadora con los datos resampleados
    resultado = carrier*data2
    #grafico normal sin resample
    graph(t, data, "Tiempo[s]", "Amplitud [db]", "Tiempo vs Amplitud")
    #grafico modulado con resample
    graph(newTime[1000:2000], resultado[1000:2000], "Tiempo[s]", "Amplitud [db]", "Tiempo vs Amplitud Modulado AM al "+title)
    #grafico normal fourier sin resample
    freqO, fourierTO = fourier(rate, info, data)
    graph(freqO, fourierTO, "Frecuencia[hz]", "Magnitud de Frecuencia[db]", "Transformada Fourier datos originales")
    #fourier resample
    freq, fourierT = fourier(rate*10, info, data2)
    #se grafica Frecuencia vs Magnitud
    graph(freq, fourierT, "Frecuencia[hz]", "Magnitud de Frecuencia[db]", "Transformada Fourier resampleada")
    #grafico fourier modulado
    freq2, fourierT2 = fourier(rate*10, info, resultado)
    #se grafica Frecuencia vs Magnitud
    graph(freq2, fourierT2, "Frecuencia[hz]", "Magnitud de Frecuencia[db]", "Frecuencia vs Magnitud de Frecuencia Modulado AM al " + title)
    return fc,  newTime, resultado, beta, data2


#Entradas: datos correspondientes a la señal modulada y resampleada
#Funcionamiento: se multiplican los datos modulados por la misma señal portadora que fue aplicada al momento de modular
#luego se aplica un filtro de paso bajo para finalmente obtener la señal original
#Salidas: se retornan los datos demodulados
def AM_demodulation(data,rate,fc, newTime, beta, info):
	#se multiplica la señal portadora original con los datos modulados
    resultado = data*np.cos(2*np.pi*fc*newTime)/beta
    #grafico demodulado
    graph(newTime, resultado, "Tiempo[s]", "Amplitud [db]", "Tiempo vs Amplitud demodulado AM")
    #grafico demodulado fourier
    freq, fourierT = fourier(rate*10, info, resultado)
    #se grafica Frecuencia vs Magnitud
    graph(freq, fourierT, "Frecuencia[hz]", "Magnitud de Frecuencia[db]", "Frecuencia vs Magnitud de Frecuencia demodulado AM")
    #se entregan los datos demodulados a la funcion firLowPass, que se encarga de realizar un filtro paso bajo
    firLowPass(10*rate, resultado, newTime, info, fc)
    return resultado

def fsk_modulation(data, rate, time, info):
	fs = rate  # sampling rate
	baud = 300  # symbol rate
	Nbits = 10**5  # number of bits
	data2 = interpolate(time, data, rate, info)
	newLen = len(data2)
	#crea un nuevo arreglo tiempo que tenga la misma dimension que el arreglo de datos
	newTime = np.linspace(0,len(data)/(rate), newLen)
	Ns = fs/baud
	N = Nbits * Ns
	f0 = 30000
	#bits = randn(Nbits,1) > 0 
	#M = np.tile(bits*2-1,(1,Ns))
	delta_f = 600
	# compute phase by integrating frequency
	ph = 2*pi*cumsum(f0 + data2*delta_f)/fs
	#t = r_[0.0:N]/fs
	FSK = sin(ph)
	frq, fourierTransform = fourier(rate, info, FSK)
	freq, fourierT = fourier(rate, info, data2)
	#fig = figure(figsize = (16,4))
	graph(newTime[1000:4000], FSK[1000:4000], "Tiempo[s]", "Amplitud[db]", "Modulacion FSK ")
	graph(freq, fourierT, "Frecuencia[hz]", "Magnitud de Frecuencia[db]", "Transformada de Fourier datos originales")
	graph(frq, fourierTransform, "Frecuencia[hz]", "Magnitud de Frecuencia[db]", "Transformada de Fourier Modulacion")
	
def compare(dmod, ndmod):
	largo = len(dmod)
	cont = 0;
	for i in range(0,largo):
		if dmod[i]!=ndmod[i]:
			cont+=1
	percent = 100*cont/largo
	print("porcentaje de errores: ")
	print(percent)
	print("#####")
	return percent
	
def ask_modulation(binA, bp):
	br = 1/bp
	f = br*10
	t2 = np.arange(bp/100, bp+bp/100, bp/100)
	print("largo array tiempo: ")
	print(len(t2))
	largo = len(t2)
	m = []
	print("largo for: ", len(binA))
	print("antes for")
	for i in range(0, 10000):
		if(binA[i]==1):
			y = 1*np.cos(2*np.pi*f*t2)
		elif(binA[i]==0):
			y=0*np.cos(2*np.pi*f*t2)
		m = np.concatenate((m,y))
	print("despues for")
	print("largo de m, señal odulada")
	print(len(m))
	t3 = np.arange(bp/100, 10000*bp+bp/100, bp/100)
	return m, largo, f, t3, bp
def awgn(lenData, SNR):
	return np.random.normal(0.0, 1.0/SNR, lenData)
def decToBinary(data):
    auxArray = []
    for i in data:
        binary = bin(int(i))[2:]
        if binary[0]=='b':
            i=i*-1
            binary = "1"+bin(int(i))[2:]
        else:
            binary = "0" + binary
        auxArray.append(binary)
    maxLen = max(auxArray, key=len)
    binArray = []
    for dato in auxArray:
        binArray.append(dato.zfill(len(maxLen)))
    return binArray
def ask_demodulation(bp, askSignal, lenT):
	mn=[]
	f=10/bp
	print("largo de for:")
	print(len(askSignal)+lenT)
	for n in range(lenT,len(askSignal)+lenT,lenT):
		t=np.arange(bp/100,bp,bp/100)
		y=np.cos(2*np.pi*f*t)                                       
		mm=np.multiply(y,askSignal[n-(lenT-1):n])
		t4=np.arange(bp/100,bp,bp/100)
		z=np.trapz(t4,mm)                                              
		zz=np.round((2*z/bp))                                     
		if(zz>=1.0):                     
			a=1
		else:
			a=0
		mn.append(a)
	return mn	
def transform_to_int(data):

	int_array = []

	for i in range(len(data)):
		#int_array2.append(list(map(int, data[i])))
		int_array +=list(map(int, data[i]))
	#int_array2 = np.array(int_array2).flatten()
	return int_array
def graficar_binario(arrayBin, bp, title):
	print(len(arrayBin))
	yArray = []
	for i in range(0, len(arrayBin)):
		for j in range(0,100):
			yArray.append(arrayBin[i])
	t = np.arange(bp/100, bp*len(yArray) + bp/100, bp/100)
	graph(t[:10000],yArray[:10000], "Tiempo", "Amplitud", "Tiempo vs Amplitud " + title) 	
	return
	
menu()

