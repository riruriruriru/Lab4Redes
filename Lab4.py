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
			binArray = decToBinary(data) #se transforman los datos a binario en forma de lista de listas 
			bin_transformed = transform_to_int(binArray) #se mapean a int y se aplana el arreglo
			tiempo = np.arange(10/100,(10)*10000+10/100,10/100) #se crea un arreglo de tiempo para 10000 bits y bp = 10
			print("Cargando...")
			mod, largo, f, tiempoMod, bp = ask_modulation(bin_transformed[:10000], 10) #se realiza la modulacion con bp = 10 y 10000 datos
			print("Listo")
			option = second_menu(rate, info, data, timp, t, bin_transformed[:10000], tiempo[:10000], mod, largo, f, tiempoMod, bp) #se entregan los datos al segundo menu

		else:
			print('Ingrese una opcion correcta')
            
	return
            
def second_menu(rate, info, data, timp, t, arrayBin, tiempo, mod, largo, f, tiempoMod, bp):
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
			graficar_binario(arrayBin,10, "Original") #se grafican los datos originales con bp iguala 10
		elif user_input=="2":
			option = third_menu_ask(rate, data, t, info, arrayBin, tiempo, mod, largo, f, tiempoMod, bp) #se ingresa al tercer menu
		elif user_input=="4":
			return 2
		else:
			print("Ingrese una opcion correcta")
	return

def third_menu_ask(rate, data,  t, info, arrayBin, tiempo, mod, largo, f, tiempoMod, bp):
	option = 0
	while option == 0:
		print('####################')
		print('Menu ASK')
		print('Opciones:')
		print('1) Aplicar modulacion ASK')
		print('2) Retroceder al menu anterior')
		print('3) Salir')
		user_input = input('Ingrese el numero de la opcion que desea ejecutar: ')
		if user_input=="1":
			graph(tiempoMod, mod, "Tiempo[s]", "Amplitud", "Modulacion ASK") #se grafica la modulada
			ASK_demodulation_menu(tiempoMod, mod,bp,largo,f, data, rate, info, arrayBin) #se ingresa al menu de demodulada
		elif user_input=="2":
			return 0
		elif user_input=="3":
			return 2
		else:
			print('Ingrese una opcion valida')
	return

def ASK_demodulation_menu(tiempo, modulatedSignal, bp, largo, f, data, rate, info, arrayBin):
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
			p=compare(demod, arrayBin)
			graphCompare(tiempo[:10000],demod[:10000],tiempo[:10000],arrayBin[:10000],"Tiempo[s]","Bits","Demodulacion sin ruido vs señal original", p, bp,0)
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
#Entrada: nombre de archivo ingresado por el usuario
#funcion que se encarga de recibir un audio y obtener sus datos
#salida: arreglo con datos, tiempo y rate
def process_audio(archivo):
    
    rate,info=wavfile.read(archivo)
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
    return rate, info, data, timp, t
#entradas: valores para eje x e y, etiquetas y titulo
#funcion que recibe dos arreglos de la misma dimension y los grafica con las etiquetas tambien recibidas por argumento
#salida: se muestra el grafico por pantalla
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
#Entradas: arreglos de tiempo y datos de las demoduladas sin y con ruido, tiempo de bit, valor del snr y titulo del grafico
#Funcionamiento: se muestran dos señales (una con ruido y otra sin ruido) repitiendo bits 100 veces hasta completar 10000 bit, se muestran dos graficos por pantalla
#Salidas: se muestra una figura con dos graficos por pantalla
def graphCompare(x, y,x2,y2, labelx, labely, title, percent, bp, snr):
	yArray = []
	yArray2 = []
	for i in range(0, 100):#se repiten 100 bits 100 veces para lograr los 10000 pedidos en la demodulada sin ruido, cambiar aca si se desean graficar mas datos
		for j in range(0,100):
			yArray.append(y[i])
	for i in range(0, 100): #se realiza lo mismo en el arreglo de la demodulada con ruido
		for j in range(0,100):
			yArray2.append(y2[i])
	t = np.arange(bp/100, bp*len(yArray) + bp/100, bp/100)
	t2 = np.arange(bp/100, bp*len(yArray2) + bp/100, bp/100)
	fig = plt.figure(1) #se crea la figura
	fig.suptitle(title + " " +str(percent)+"%", fontsize=16) #se asigna titulo a la figura           
	ax = plt.subplot(211)             #primer grafico de la figura
	ax.set_title("Demodulada sin ruido")
	plt.plot(x[:10000],yArray[:10000]) #datos de la demodulada sin ruido
	ax = plt.subplot(212)             #segundo grafico
	plt.plot(x2[:10000],yArray2[:10000]) #datos de la demodulada con ruido
	plt.show() #se muestra el grafico
	return
#Entradas: señal demodulada sin ruido y con ruido
#Funcionamiento: se comparan las señales valor por valor y se cuentan los errores
#Salida: valor que representa el porcentaje de error
def compare(dmod, ndmod):
	largo = len(dmod) #largo de la señal sin ruido, ambas son del mismo tamaño
	cont = 0;
	for i in range(0,largo):
		if dmod[i]!=ndmod[i]: #se cuentan los errores
			cont+=1
	percent = 100*cont/largo #se pasa el valor a porcentaje
	print("porcentaje de errores: ")
	print(percent)
	print("#####")
	return percent #se retorna el porcentaje
#Entradas: arreglo de binarios y tiempo de bit
#funcionamiento: se modula multiplicando la portadora por 1 o 0, debido a que la amplitud no se modifica, se concatenan estos valores generando un arreglo con los valores modulados
#Salida: arreglo de modulada, largo del arreglo de tiempo generado, frecuencia, arreglo de tiempo con la misma dimension que la modulada y tiempo de bit
def ask_modulation(binA, bp):
	br = 1/bp #bit rate
	f = br*10 #frecuencia
	t2 = np.arange(bp/100, bp+bp/100, bp/100) #arreglo de tiempo para la portadora
	largo = len(t2) #largo del arreglo
	mod = [] #arreglo de la modulada
	for i in range(0, 10000): #ciclo para modular 10000 bits
		if(binA[i]==1): #si el valor binario es igual a uno:
			carrier = 1*np.cos(2*np.pi*f*t2) #se multiplica 1 por la portadora
		elif(binA[i]==0):
			carrier=0*np.cos(2*np.pi*f*t2) #en caso contrario, se multiplica por 0
		mod = np.concatenate((mod,carrier)) #se concatenan los valores de la portadora con la señal modulada 
	t3 = np.arange(bp/100, 10000*bp+bp/100, bp/100) #arreglo de tiempo con la misma dimension que la señal modulada
	return mod, largo, f, t3, bp
#Entradas: snr (signal to noise ratio) y largo del arreglo al cual se le desea añadir ruido
#Procedimiento: se genera un arreglo de valores aleatorios con distribucion normal del mismo largo que los datos a los cuales se les quiere añadir ruido, ademas se aplica el valor de snr
#Salidas: arreglo de ruido blanco aditivo
def awgn(lenData, SNR):
	return np.random.normal(0.0, 1.0/SNR, lenData)
#Entradas: arreglo original de datos del archivo de audio
#Funcionamiento: se recorre el arreglo e datos y se castea su contenido a binarios, agregando el bit de signo correspondiente y eliminando los caracteres de formato binario 0b, generandose un arreglo de arreglos
#Salida: lista de listas de binarios en formato str
def decToBinary(data):
    auxArray = []
    for i in data: #se recorren los datos
        binary = bin(int(i))[2:] #se castea el valor a int eliminando los caracteres 0b
        if binary[0]=='b': #si aun existe una b, significa que el numero era de valor negativo
            binary = "1"+bin(int(abs(i)))[2:] #se transforma nuevamente el valor pero en formato positivo y agregando el bit "1" que representa signo positivo
        else:
            binary = "0" + binary #si el valor ya es positivo, se le agrega el bit 0 inicial
        auxArray.append(binary) #se agrega el binario al arreglo auxiliar
    maxLen = max(auxArray, key=len) #se busca el largo maximo en el arreglo auxiliar
    binArray = []
    for elem in auxArray: #se recorre el arreglo auxiliar
        binArray.append(elem.zfill(len(maxLen))) #todos los elementos se igualan a la misma longitud agregando ceros a la izquierda
    return binArray
#Entradas: tiempo de bit, señal ask binaria y largo del arreglo de tiempo generado en la modulacion
#Funcionamiento: Se multiplica la portadora con trozos de la señal modulada y luego se integra, comparando el valor con la amplitud esperada = 1 y asignando valores de bit iguales a 1 o 0
#Salidas: arreglo de señal demodulada
def ask_demodulation(bp, askSignal, lenT):
	demod=[] #arreglo que guarda los datos de la demodulada
	f=10/bp #frecuencia segun tiempo de bit
	#ciclo for en base al largo del arreglo de tiempo generado en la modulacion
	t=np.arange(bp/100,bp,bp/100) #arreglo t segun el tiempo de bit
	for n in range(lenT,len(askSignal)+lenT,lenT):
		carrier=np.cos(2*np.pi*f*t)   #funcion portadora                                     
		aux=np.multiply(carrier,askSignal[n-(lenT-1):n]) #se multiplica la portadora con un trozo de la señal modulada
		integrate=np.trapz(t,aux) # integral                                              
		rounded=np.round((2*integrate/bp))   #se redondea la division entre el valor de la integral y el tiempo de bit                       
		if(rounded>=1.0): #como se espera una amplitud de 1 (debido a que la modulacion ASK no modifico la amplitud), si el valor de la integral sobrepasa 1, se asigna un valor de bit = 1                     
			a=1
		else: #caso contrario, bit = 0
			a=0
		demod.append(a) #se agrega al arreglo demodulado
	return demod
#Entrada: arreglo de datos binarios	en forma de lista de listas
#Funcionamiento: recorre la lista de listas y mapea cada una de las sublistas concatenandolas entre si (dejando el arreglo plano) y transformando cada bit de str a int
#Retorno: arreglo de datos binario plano y con datos de tipo int
def transform_to_int(aBin):
	int_array = [] #arreglo nuevo
	for i in range(len(aBin)): #for para todo el arreglo de datos
		int_array +=list(map(int, aBin[i])) #se mapean todos los elementos del sub arreglo a array y se concatena en el array vacio 
	return int_array #se retorna el nuevo arreglo de bits
#Entrada: tiempo de bit, titulo y arreglo de binarios
#Funcionamiento: repite bits del arreglo 100 veces para visualizar la señal como pulsos cuadrados, se grafican 10000 datos
#Salida: se muestra por pantalla un grafico del arreglo remuestrado
def graficar_binario(arrayBin, bp, title):
	print(len(arrayBin))
	yArray = []
	for i in range(0, 100): #como se muestran 10000 bits y cada bit se debe repetir 100 veces, solo se recorren los 100 primeros bits del arreglo
		for j in range(0,100): #cada bit se repite 100 veces
			yArray.append(arrayBin[i]) #se agregan a un nuevo arreglo
	t = np.arange(bp/100, bp*len(yArray) + bp/100, bp/100) #se crea un arreglo de tiempo con la misma dimension que el arreglo de bit nuevo
	graph(t[:10000],yArray[:10000], "Tiempo", "Amplitud", "Tiempo vs Amplitud " + title) 	 #se grafica tiempo vs amplitud
	return
	
menu()

