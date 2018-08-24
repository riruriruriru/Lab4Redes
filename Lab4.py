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
            #print("array binarios: ")
            #print(binArray)
            binFlat = transform_to_int(binArray)
            #print("ARREGLO BINARIO FLATTENED")
            #print(binFlat)
            ask_modulation(binFlat[:100], 10)
            option = second_menu(rate, info, data, timp, t)
            
        else:
            print('Ingrese una opcion correcta')
            
    return
            
def second_menu(rate, info, data, timp, t):
    option = 0
    #Segundo menu que se llama cuando se abre correctamente un archivo
    while option == 0:
        print('##########################')
        print('Menu Archivo')
        print('Opciones:')
        print('1) Modulacion FM analoga')
        print('2) Modulacion AM analoga')
        print('3) Retroceder al menu anterior')
        print('4) Salir')
        user_input = input('Ingrese el numero de la opcion que desea ejecutar: ')
        if user_input=="3":
            #opcion que sirve para retroceder al menu anterior y abrir un nuevo archivo
            return 0
        elif user_input=="1":            
            option = third_menu(rate, data, t, info)
        elif user_input=="2":
            option = third_menu_am(rate, data, t, info)
        elif user_input=="4":
            
            return 2
        else:
            print("Ingrese una opcion correcta")
    return


def third_menu(rate, data, t, info):
    option = 0
    while option == 0:
        print('####################')
        print('Menu FM')
        print('Opciones:')
        print('1) Aplicar modulacion analoga al 15%')
        print('2) Aplicar modulacion analoga al 100%')
        print('3) Aplicar modulacion analoga al 125%')
        print('4) Retroceder al menu anterior')
        print('5) Salir')
        user_input = input('Ingrese el numero de la opcion que desea ejecutar: ')
        if user_input=="1":
            FM_analog_modulation(rate, data, 0.15, t, info)
            #option = fourth_menu(rate, f_data, f_data2, f_data3, "Low Pass")
        elif user_input=="2":
            FM_analog_modulation(rate, data, 1, t, info)
            #option = fourth_menu(rate, f_data, f_data2, f_data3, "High Pass")                        
        elif user_input=="3":
            FM_analog_modulation(rate, data, 1.25, t, info)
            #option = fourth_menu(rate, f_data, f_data2, f_data3, "Band Pass")                 
        elif user_input=="4":
            return 0
        elif user_input=="5":
            return 2
        else:
            print('Ingrese una opcion valida')
    return
def third_menu_am(rate, data,  t, info):
    option = 0
    while option == 0:
        print('####################')
        print('Menu AM')
        print('Opciones:')
        print('1) Aplicar modulacion analoga al 15%')
        print('2) Aplicar modulacion analoga al 100%')
        print('3) Aplicar modulacion analoga al 125%')
        print('4) Retroceder al menu anterior')
        print('5) Salir')
        user_input = input('Ingrese el numero de la opcion que desea ejecutar: ')
        if user_input=="1":
            fc,  newTime, resultado, beta, data2 = AM_analog_modulation(rate, data, 0.15, t, info)
            AM_demodulation_menu(fc,  newTime, resultado, beta, data2, rate, info)
        elif user_input=="2":
            fc,  newTime, resultado, beta, data2 = AM_analog_modulation(rate, data, 1, t, info)
            AM_demodulation_menu(fc,  newTime, resultado, beta, data2, rate, info)
        elif user_input=="3":
            fc,  newTime, resultado, beta, data2 = AM_analog_modulation(rate, data, 1.25, t, info)
            AM_demodulation_menu(fc,  newTime, resultado, beta, data2, rate, info)
        elif user_input=="4":
            return 0
        elif user_input=="5":
            return 2
        else:
            print('Ingrese una opcion valida')
    return

#Filtro de tipo fir low pass
def firLowPass(rate, data, t, info, fc):
    #se calcula la frecuencia de nyquist 
    nyq_f = rate/2.0
    cutoff =  3000/nyq_f
    numtaps = 1001
    print("###########################")
    print("nyq: ")
    print(nyq_f)
    #se obtienen los valores que se usaran para filtrar con firwin
    print("cutoff: ")
    print(cutoff/nyq_f)
    print("#########")
    taps = signal.firwin(numtaps, 3000/nyq_f, window = 'hamming')
    #se aplica el filtro con lfilter
    t2 = linspace(0,len(data)/(rate),len(data))
    y = signal.lfilter(taps, 1.0, data)
    #se grafican los tres filtros
    freq, fourierT = fourier(rate, info, y)
    graph(freq, fourierT, "Frecuencia[hz]", "Magnitud de Frecuencia[db]","Transformada de Fourier: filtro Low-Pass señal demodulada")
    #se retornan los valores de las amplitudes
    return y


def AM_demodulation_menu(fc,  newTime, resultado, beta, data, rate, info):
	option = 0
	text = str(beta*100) + "%"
	while option == 0:
		print('####################')
		print('Menu AM Demodulacion')
		print('Opciones:')
		print('1) Aplicar demodulacion analoga AM al: ' + text)
		print('2) Retroceder al menu anterior')
		print('3) Salir')
		user_input = input('Ingrese el numero de la opcion que desea ejecutar: ')
		if user_input=="1":
			print("fc: ", fc)
			AM_demodulation(resultado, rate, fc, newTime, beta, info)
		elif user_input=="2":
			return 0                        
		elif user_input=="3":
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
    plt.title(title)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.plot(x, y)
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
	
algo = {
	"id":"dasd",
	

}
def ask_modulation(binA, bp):
	A1 = 1
	A2 = 0
	br = 1/bp
	f = br*10
	t2 = np.arange(bp/99, bp+bp/99, bp/99)
	print("largo array tiempo: ")
	print(len(t2))
	largo = len(t2)
	m = []
	print("largo for: ", len(binA))
	print("antes for")
	for i in range(0, len(binA)):
		if(binA[i]==1):
			y = A1*np.cos(2*np.pi*f*t2)
		elif(binA[i]==0):
			y=A2*np.cos(2*np.pi*f*t2)
		m = np.concatenate((m,y))
	print("despues for")
	print("largo de m, señal odulada")
	print(len(m))
	t3 = np.arange(bp/99, len(binA)*bp+bp/99, bp/99)
	graph(t3, m, "Tiempo[s]", "Amplitud[db]", "Modulacion ASK ")
	return m, largo, f
def awgn(lenData, SNR):
	np.random.normal(0.0, 1.0/SNR, lenData)
def decToBinary(data):
    auxArray = []
    for i in data:
        binary = bin(i)[2:]
        if binary[0]=='b':
            i=i*-1
            binary = "1"+bin(i)[2:]
        else:
            binary = "0" + binary
        auxArray.append(binary)
    maxLen = max(auxArray, key=len)
    binArray = []
    for dato in auxArray:
        binArray.append(dato.zfill(len(maxLen)))
    return binArray
def ask_demodulation(bp, askSignal, lenT):
	mn=[];
	f=10/bp
	for n in range(lenT, length(askSignal)+lenT, lenT):
		t=np.arange(bp/99,bp,bp/99)
		carrier=np.cos(2*np.pi*f*t)                                        # carrier siignal 
		mm=np.multiply(carrier,askSignal[n-(lenT-1):n])
		t4=np.arange(bp/99,bp,bp/99)
		z=trapz(t4,mm)                                              # intregation 
		zz=np.round((2*z/bp))                                     
		if zz>7.5:                                  # logic level = (A1+A2)/2=7.5
			a=1
		else:
			a=0
		mn=mn.append(a)
	return mn
def transform_to_int(data):
    
    int_array = []
    print("DENTRO FLAT")
    print(data[0])
    print(data[1])
    print(data[1][7])
    print(int("10011101"))
    print(list("{0:b}".format(-37)))
    print(list("{0:b}".format(37)))
    a = list("{0:b}".format(-37))
    print(list(map(int, list("{0:b}".format(37)))))
    print(list(map(int, a)))
    print(list(map(int, list("{0:b}".format(37)))).append(list(map(int, list("{0:b}".format(-37))))))
    for i in range(len(data)):
        for j in range(len(data[i])):
            int_array.append(int(data[i][j]))
    return int_array
menu()

