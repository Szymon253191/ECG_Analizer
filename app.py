import os
from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import wfdb
import Pmw
import json
from ecgdetectors import Detectors
from scipy import signal
from PIL import Image, ImageTk

# inicjacja tkintera
master = Tk() 

#inicjacja Pwm
Pmw.initialise(master)

# wgranie ikony 
ico = Image.open("icon.bmp")
photo = ImageTk.PhotoImage(ico)
master.wm_iconphoto(False, photo)

# ustalenie wymiarow okna
master.geometry("800x600")
master.resizable(0, 0)

# nazwa okna
master.title("Program porownujacy metody analizy sygnalu EKG")

# stworzenie obiektu dymkow po najechaniu na przycisk
tooltip1 = Pmw.Balloon(master)
lbl = tooltip1.component("label")
lbl.config(background="white", foreground="black")

# zaladownie zdjecia tla aplikacji
backgroundImage = PhotoImage(file="BackgroundPhoto1600x700.gif")
backgroundImageLabel = Label(master,
                             image=backgroundImage)
backgroundImageLabel.place(x=-1, y=-1)


def ChooseFile():
    """
    Funkcja otwierajaca okno wyszukiwania pliku. Po wybraniu sciezka jest dzielona czesci 
    a sama nazwa pliku jest wstawiana do zmiennej globalnej.
    """
    filetypes = (
        ('dat files', '*.dat'),
        ('atr files', '*.atr'),
        ('All files', '*.*')
    )

    global filePath
    filePath = filedialog.askopenfilename(title='Wybierz plik',
                                          filetypes=filetypes)

    global filePathNoDot
    filePathNoDot, _, _ = filePath.partition('.')

    filename = os.path.basename(filePath)
    global fileNoDot
    fileNoDot, _, _ = filename.partition('.')

    pathToFile = Label(master,
                       height=1,
                       width=25,
                       text="Wybrany plik: "+fileNoDot,
                       background="white")
    pathToFile.place(x=450, y=228)

def GetRecord(file):
    """
    Funkcja sciagajaca dane z bazy wfdb na temat pliku oraz sprawdza poprawnosc danych 
    sygnalow zapisanych na komputerze.
    
    Args:
        file (str): wartosc zapisana w typie string zawierajaca nazwe pliku.

    Returns:
        record (numpy.ndarray): szereg wartosci zawierajacy wartosc sygnalu dla kazdej probki.
        fields (dict): zbior danych zawierajacy informacje o sygnale.
        annotation (wfdb.io.annotation.Annotation): zbior danych dostarczony w paczce wfdb 
            zawierajacy informacje o sygnale.
    """
    record, fields = wfdb.rdsamp(file,channels=[0])
    annotation = wfdb.rdann(file, 'atr')
    
    return record, fields, annotation


def PlotSignal():
    """
    Funkcja wyswietlajaca sygnal
    """
    record, ax, annotation = GetRecord(filePathNoDot)
    plt.clf()
    plt.plot(record, linewidth=0.5,color='k')
    plt.minorticks_on()

    plt.grid(which='major', linestyle=':', color='red', linewidth='1.0', alpha=0.7)
    plt.grid(which='minor', linestyle=':', color='red', linewidth='0.5', alpha=0.5)
    # plt.plot(annotation.sample, record[annotation.sample], 'ro')
    plt.xlabel("Numer probki")
    plt.ylabel("Amplituda [mV]")
    plt.title("Sygnal: "+fileNoDot
            #   + " oraz QRS z pliku .ann"
              )
    
    plt.show()

def BandPassECG(record,Fs,high_pass_cutt_off=5,low_pass_cutt_off=15):
    """
    Funkcja stosujaca filtr Butterwortha 4 rzedu na sygnale przy podanej wartosci 
    czestotliwosci probkowania sygnalu.

    Args:
        record (array): szereg wartosci zawierajacy wartosc sygnalu dla kazdej probki
        Fs (int): czestotliwosc sygnalu poddawanego filtracji.
        high_pass_cutt_off (int, optional): dolny prog odciecia filtra pasmowoprzepustowego. 
            Defaults to 5.
        low_pass_cutt_off (int, optional): gorny prog odciecia filtra pasmowoprzepustowego. 
            Defaults to 15.

    Returns:
        ECG_BP (numpy.ndarray): szereg wartosci zawierajacy wartosci sygnalu po zastosowaniu 
        filtra pasmowo-przepustowego dla kazdej probki.
        ECG (numpy.ndarray): szereg wartosci zawierajacy wartosc sygnalu dla kazdej probki.
    """
    ECG    = record
    W1     = high_pass_cutt_off*2/Fs                 
    W2     = low_pass_cutt_off*2/Fs                  
    b, a   = signal.butter(4, [W1,W2], 'bandpass')   
    ECG    = np.asarray(ECG)                         
    ECG    = np.squeeze(ECG)                         

    ECG_BP = signal.filtfilt(b,a,ECG) 

    return ECG_BP,ECG

def BandpassPreAlgorithm(record, freq, lowcut = 0.5, highcut=40):
    """
    Funkcja stosujaca filtr Butterwortha 4 rzedu na sygnale przy podanej wartosci 
    czestotliwosci probkowania sygnalu.

    Args:
        record (array): szereg wartosci zawierajacy wartosc sygnalu dla kazdej probki
        freq (int): czestotliwosc sygnalu poddawanego filtracji
        lowcut (int, optional): dolny prog odciecia filtra pasmowoprzepustowego. 
            Defaults to 0.5.
        highcut (int, optional): gorny prog odciecia filtra pasmowoprzepustowego. 
            Defaults to 40.

    Returns:
        ecg_filtered (numpy.ndarray): szereg wartosci zawierajacy wartosci sygnalu 
            po zastosowaniu filtra pasmowo-przepustowego dla kazdej probki.
    """
    # Create a Butterworth bandpass filter.
    b, a = signal.butter(4, [lowcut, highcut], btype='band',fs=freq)

    # Apply the filter to your ECG signal.
    ecg_filtered = signal.lfilter(b, a, record)

    return ecg_filtered

def detect_rr(rpeaks):
    """
    Funkcja tworzaca szereg roznic odlegosci pomiedzy punktami.

    Args:
        rpeaks (list): szereg wartosci punktow w czasie.

    Returns:
        listOfR2 (list): szereg wartosci odleglosci pomiedzy punktami w czasie.
    """
    listOfR1 = np.diff(rpeaks)
    listOfR2 = listOfR1.tolist()

    return listOfR2

def resample_distances(distances, new_size):
    """
    Funkcja interpolacjo-podobna tworzaca szereg wartosci zachowujacy odleglosci 
    pomiedzy punktami. Wartosc punktu wplywa na dlugosc wystepowania tej wartosci w szeregu.

    Args:
        distances (list): szereg wartosci odleglosci pomiedzy punktami w czasie.
        new_size (list): dlugosc do ktorej szereg wartosci ma zostac rozciagniety.

    Returns:
        resampled_distances (list): Szereg wynikowy.
    """
    distances = np.array(distances)
    total_distance = distances.sum()
    proportions = distances / total_distance
    new_distances = proportions * new_size
    num_points = np.round(new_distances).astype(int)

    resampled_distances = []
    for distance, num in zip(distances, num_points):
        for _ in range(num):
            resampled_distances.append(distance)    

    return resampled_distances

def Signal_and_bmp_plot():
    """
    Funkcja wykonuje pobranie sygnalu oraz danych z pliku .ann, nastepnie wykrywa punkty 
    uderzenia serca i zapisuje je do zmiennej. W kolejnym kroku funkcja tworzaca szereg
    roznic odleglosci tworzy nowa zmienna, ktora jest rozciagana za pomoca funkcji
    "rozciagajacej". Finalnie sygnal i wynik rozciagania jest wyswietlany na wykresie
    a miejsca wpisujace sie w kryteria odchodzace od norm zostaja uwidocznione.
    """
    record, fields, annotation = GetRecord(filePathNoDot)
    Fs = int(fields['fs'])
    _,ECG_raw = BandPassECG(record,Fs)
    
    RR_time_BMP = detect_rr(annotation.sample)
    RR_time_BMP[:] = [x / Fs for x in RR_time_BMP]


    valY = resample_distances(RR_time_BMP,len(ECG_raw))
    valY.extend([valY[-1]] * (len(ECG_raw) - len(valY)))
    
    # old_y_len = np.arange(0,len(RR_time_BMP))
    # y_lin = np.linspace(0,len(RR_time_BMP),len(ECG_raw))
    # spl = UnivariateSpline(old_y_len,RR_time_BMP,k=3,s=3)
    # new_y = spl(y_lin)
    valX = np.arange(len(valY))
    # print(new_y)

    fig, ax = plt.subplots(2,1,sharex=True)
    tach_tresh = 0.6
    brad_tresh = 1


    ax[0].plot(ECG_raw,linewidth=0.5,color='k')
    ax[0].plot(annotation.sample, record[annotation.sample], 'ro')
 
    ax[0].minorticks_on()
    ax[0].set_title("Sygnal: "+fileNoDot)

    # ax[0].set_xlabel("Numer prďż˝bki")
    ax[0].set_ylabel("Amplituda [mV]")
    
    ax[1].set_title("Autorska metoda analizy arytmii")
        
    ax[1].set_xlabel("Numer probki")
    ax[1].set_ylabel("Odleglosc miedzy\nzespolami QRS [ms]")
    
    # Make the major grid
    ax[0].grid(which='major', linestyle=':', color='red', linewidth='1.0', alpha=0.7)
    # Make the minor grid
    ax[0].grid(which='minor', linestyle=':', color='red', linewidth='0.5', alpha=0.5)

    ax[1].grid()
    ax[1].plot(np.array(valY),linewidth=0.8,alpha=1,color='k')

    ax[1].fill_between(valX,0,
                        np.array(valY).max(),
                        where=np.array(valY) > brad_tresh, 
                        color='b',
                        alpha=0.3)

    ax[1].fill_between(valX,0,
                        np.array(valY).max(),
                        where=np.array(valY) < tach_tresh, 
                        color='r',
                        alpha=0.3)

    ax[1].fill_between(valX[:-1],0,
                        np.array(valY).max(),
                        where=(
                            (np.array(valY)[1:]) < np.array(valY)[:-1]-np.array(valY)[:-1]*.15), 
                        color='y',
                        alpha=0.7)

    ax[1].fill_between(valX[:-1],0,
                        np.array(valY).max(),
                        where=(
                            (np.array(valY)[1:]) > np.array(valY)[:-1]+np.array(valY)[:-1]*.15),
                        color='y',
                        alpha=0.7)

    ax[1].axhline(np.mean(RR_time_BMP), 
                  linestyle='dashed', 
                  color='xkcd:dark grey',
                  alpha=0.6,
                  label='Srednie tempo', 
                  marker='')
    ax[1].legend()
    # ax[0].set_xlim(1000, Fs*10)
    # fig = ax[0].get_figure()
    # resetax = fig.add_subplot(2,3,4)
    # button = Button(resetax, 'Reset', hovercolor='0.975')
    # button.on_clicked(lambda event: ax[0].set_xlim(0, 100))
    
    plt.show()

def PanTompkins():
    """
    Funkcja dokonujaca pobrania sygnalu oraz analizy sygnalu algorytmem Pan-Tompkins. Wynik 
    analizy jest od razy wyswietlany.

    Returns:
        rpeaks (list): szereg wartosci punktow w czasie.
    """
    record, fields, _ = GetRecord(filePathNoDot)
    Fs = int(fields['fs'])
    plt.clf()

    detectors = Detectors(Fs)
    # betterRecord = BandpassPreAlgorithm(record,Fs,0.5,40)
    # plt.plot(betterRecord)
    # plt.show()
    r_peaks = detectors.pan_tompkins_detector(record[:,0])

    plt.plot(record, linewidth=0.5,color='k')
    plt.minorticks_on()

    # Make the major grid
    plt.grid(which='major', linestyle=':', color='red', linewidth='1.0', alpha=0.7)
    # Make the minor grid
    plt.grid(which='minor', linestyle=':', color='red', linewidth='0.5', alpha=0.5)
    plt.plot(r_peaks, record[r_peaks], 'ro')

    plt.title("Sygnal: "+fileNoDot+" oraz QRS z analizy algorytmem PanTompkins")
    plt.xlabel("Numer probki")
    plt.ylabel("Amplituda [mV]")
    plt.show()

    return r_peaks

def Hamilton():
    """
    Funkcja dokonujaca pobrania sygnalu oraz analizy sygnalu algorytmem Hamilton. 
    Wynik analizy jest od razy wyswietlany.

    Returns:
        r_peaks (list): szereg wartosci punktow w czasie.
    """
    record, fields, _ = GetRecord(filePathNoDot)
    Fs = int(fields['fs'])
    detectors = Detectors(Fs)
    r_peaks = detectors.hamilton_detector(record[:,0])
    plt.clf()
    plt.plot(record, linewidth=0.5,color='k')
    plt.minorticks_on()

    # Make the major grid
    plt.grid(which='major', linestyle=':', color='red', linewidth='1.0', alpha=0.7)
    # Make the minor grid
    plt.grid(which='minor', linestyle=':', color='red', linewidth='0.5', alpha=0.5)
    plt.plot(r_peaks, record[r_peaks], 'ro')
    plt.xlabel("Numer probki")
    plt.ylabel("Amplituda [mV]")

    plt.title("Sygnal: "+fileNoDot+" oraz QRS z analizy algorytmem Hamilton")
    plt.show()

    return r_peaks

def SWT():
    """
    Funkcja dokonujaca pobrania sygnalu oraz analizy sygnalu algorytmem SWT. 
    Wynik analizy jest od razy wyswietlany.

    Returns:
        r_peaks (list): szereg wartosci punktow w czasie.
    """
    record, fields, _ = GetRecord(filePathNoDot)
    Fs = int(fields['fs'])
    detectors = Detectors(Fs)
    r_peaks = detectors.swt_detector(record[:,0])
    plt.clf()
    plt.plot(record, linewidth=0.5,color='k')
    plt.minorticks_on()

    # Make the major grid
    plt.grid(which='major', linestyle=':', color='red', linewidth='1.0', alpha=0.7)
    # Make the minor grid
    plt.grid(which='minor', linestyle=':', color='red', linewidth='0.5', alpha=0.5)
    plt.plot(r_peaks, record[r_peaks], 'ro')
    plt.xlabel("Numer probki")
    plt.ylabel("Amplituda [mV]")

    plt.title("Sygnal: "+fileNoDot+" oraz QRS z analizy algorytmem SWT")
    plt.show()

    return r_peaks

def AnalizeSignalPT():
    """
    Funkcja dokonuje pobranie sygnalu i analizuje go algorytmem Pan-Tompkins. 
    Wynikiem sa informacje o wyniku analizy.

    Returns:
        text1 (str): informacja o srednim tempie bicia serca dla wynikow algorytmu.
        text2 (str): informacja o sredniej odleglosci pomiedzy wykrytymi uderzeniami.
        text3 (str): informacja o ilosci znalezionych uderzen w calym sygnale.
        text4 (str): informacja o srednim tempie bicia serca dla pliku .ann.
        text5 (str): informacja o sredniej odleglosci pomiedzy uderzeniami z pliku .ann.
        text6 (str): informacja o ilosci znalezionych uderzen w pliku .ann.
        text7 (str): informacja o roznicy wykrywych uderzen a tych znajdujacych sie w pliku .ann.
    """
    record, fields, annotation = GetRecord(filePathNoDot)
    Fs = int(fields['fs'])
    
    detectors = Detectors(Fs)
    r_peaks = detectors.pan_tompkins_detector(record[:,0])

    # RR time
    RR_time = detect_rr(r_peaks)

    text1 = ("BMP: %s" % str(round(np.mean(RR_time)/Fs*60, 2)))
    text2 = ("Mean RR (samp): %s" % str(round(np.mean(RR_time),2)))
    text3 = ("QRS Found: %s" % str(len(r_peaks)))

    RR_time_ann = detect_rr(annotation.sample)

    text4 = ("ANN BMP: %s" % str(round(np.mean(RR_time_ann)/Fs*60, 2)))
    text5 = ("Mean RR (samp): %s" % str(round(np.mean(RR_time_ann),2)))
    text6 = ("ANN QRS Found: %s" % str(len(annotation.sample)))

    text7 = ("Ann - Detected QRS: %s" % str(len(RR_time_ann)-len(r_peaks)))

    return text1, text2, text3, text4, text5, text6, text7

def AnalizeSignalH():
    """
    Funkcja dokonuje pobranie sygnalu i analizuje go algorytmem Hamilton. 
    Wynikiem sa informacje o wyniku analizy.

    Returns:
        text1 (str): informacja o srednim tempie bicia serca dla wynikow algorytmu.
        text2 (str): informacja o sredniej odleglosci pomiedzy wykrytymi uderzeniami.
        text3 (str): informacja o ilosci znalezionych uderzen w calym sygnale.
        text4 (str): informacja o srednim tempie bicia serca dla pliku .ann.
        text5 (str): informacja o sredniej odleglosci pomiedzy uderzeniami z pliku .ann.
        text6 (str): informacja o ilosci znalezionych uderzen w pliku .ann.
        text7 (str): informacja o roznicy wykrywych uderzen a tych znajdujacych sie w pliku .ann.
    """
    record, fields, annotation = GetRecord(filePathNoDot)
    Fs = int(fields['fs'])
    
    detectors = Detectors(Fs)
    r_peaks = detectors.hamilton_detector(record[:,0])

    # RR time
    RR_time = detect_rr(r_peaks)

    text1 = ("BMP: %s" % str(round(np.mean(RR_time)/Fs*60, 2)))
    text2 = ("Mean RR (samp): %s" % str(round(np.mean(RR_time),2)))
    text3 = ("QRS Found: %s" % str(len(r_peaks)))

    RR_time_ann = detect_rr(annotation.sample)

    text4 = ("ANN BMP: %s" % str(round(np.mean(RR_time_ann)/Fs*60, 2)))
    text5 = ("Mean RR (samp): %s" % str(round(np.mean(RR_time_ann),2)))
    text6 = ("ANN QRS Found: %s" % str(len(annotation.sample)))

    text7 = ("Ann - Detected QRS: %s" % str(len(RR_time_ann)-len(r_peaks)))
    
    return text1, text2, text3, text4, text5, text6, text7

def AnalizeSignalSWT():
    """
    Funkcja dokonuje pobranie sygnalu i analizuje go algorytmem SWT. 
    Wynikiem sa informacje o wyniku analizy.

    Returns:
        text1 (str): informacja o srednim tempie bicia serca dla wynikow algorytmu.
        text2 (str): informacja o sredniej odleglosci pomiedzy wykrytymi uderzeniami.
        text3 (str): informacja o ilosci znalezionych uderzen w calym sygnale.
        text4 (str): informacja o srednim tempie bicia serca dla pliku .ann.
        text5 (str): informacja o sredniej odleglosci pomiedzy uderzeniami z pliku .ann.
        text6 (str): informacja o ilosci znalezionych uderzen w pliku .ann.
        text7 (str): informacja o roznicy wykrywych uderzen a tych znajdujacych sie w pliku .ann.
    """
    record, fields, annotation = GetRecord(filePathNoDot)
    Fs = int(fields['fs'])
    
    detectors = Detectors(Fs)
    r_peaks = detectors.swt_detector(record[:,0])

    # RR time
    RR_time = detect_rr(r_peaks)

    text1 = ("BMP: %s" % str(round(np.mean(RR_time)/Fs*60, 2)))
    text2 = ("Mean RR (samp): %s" % str(round(np.mean(RR_time),2)))
    text3 = ("QRS Found: %s" % str(len(r_peaks)))

    RR_time_ann = detect_rr(annotation.sample)

    text4 = ("ANN BMP: %s" % str(round(np.mean(RR_time_ann)/Fs*60, 2)))
    text5 = ("Mean RR (samp): %s" % str(round(np.mean(RR_time_ann),2)))
    text6 = ("ANN QRS Found: %s" % str(len(annotation.sample)))

    text7 = ("Ann - Detected QRS: %s" % str(len(RR_time_ann)-len(r_peaks)))

    return text1, text2, text3, text4, text5, text6, text7

def ExportCSV_PT():
    """
    Funkcja dokonuje pobranie sygnalu, dokonuje detekcja za pomoca algorytmu 
    Pan-Tompkins, a nastepnie eksportuje szereg punktow detekcji do pliku.
    """
    record, fields, _ = GetRecord(filePathNoDot)
    Fs = int(fields['fs'])
    
    detectors = Detectors(Fs)
    r_peaks = detectors.pan_tompkins_detector(record[:,0])

    # Convert the list to a JSON string
    json_string = json.dumps(r_peaks)

    # Write the JSON string to a file
    with open(fileNoDot+"PanTomkins.csv", 'w') as outfile:
        outfile.write(json_string)
        
    # np.savetxt(fileNoDot+"PanTomkins.csv",r_peaks,delimiter=',')
    
def ExportCSV_H():
    """
    Funkcja dokonuje pobranie sygnalu, dokonuje detekcja za pomoca algorytmu 
    Hamilton, a nastepnie eksportuje szereg punktow detekcji do pliku.
    """
    record, fields, _ = GetRecord(filePathNoDot)
    Fs = int(fields['fs'])
    
    detectors = Detectors(Fs)
    r_peaks = detectors.hamilton_detector(record[:,0])
    
    json_string = json.dumps(r_peaks)

    # Write the JSON string to a file
    with open(fileNoDot+"Hamilton.csv", 'w') as outfile:
        outfile.write(json_string)
    # np.savetxt(fileNoDot+"Hamilton.csv",r_peaks,delimiter=',')
    
def ExportCSV_SWT():
    """
    Funkcja dokonuje pobranie sygnalu, dokonuje detekcja za pomoca algorytmu 
    SWT, a nastepnie eksportuje szereg punktow detekcji do pliku.
    """
    record, fields, ann = GetRecord(filePathNoDot)
    Fs = int(fields['fs'])
    
    detectors = Detectors(Fs)
    r_peaks = detectors.swt_detector(record[:,0])
    
    json_string = json.dumps(r_peaks)

    # Write the JSON string to a file
    with open(fileNoDot+"SWT.csv", 'w') as outfile:
        outfile.write(json_string)
    
    # np.savetxt(fileNoDot+"SWT.csv",r_peaks,delimiter=',')
    # np.savetxt(fileNoDot+"Ann.csv",ann.sample,delimiter=',')

def ExportCSV_Ann():
    """
    Funkcja dokonuje pobranie sygnalu oraz informacji z pliku .ann a nastepnie 
    eksportuje szereg punktow detekcji do pliku.
    """
    _, _, ann = GetRecord(filePathNoDot)
    
    json_string = json.dumps(ann.sample.tolist())

    # Write the JSON string to a file
    with open(fileNoDot+"Ann.csv", 'w') as outfile:
        outfile.write(json_string)
        
    # np.savetxt(fileNoDot+"Ann.csv",ann.sample,delimiter=',')
    
    
    
class DisplayDiffPT(Toplevel):
    """
    Klasa tworzaca nowe okno, w ktorym otwierany jest canvas.

    """

    def __init__(self, master=None):

        super().__init__(master=master)
        self.title("PanTompkins: " + fileNoDot)
        self.geometry("300x200")

        label = Label(self,
                      text="Informacje o analizie sygnalu",)
        label.pack()
        self.resizable(0, 0)
        
        a,b,c,d,e,f,g = AnalizeSignalPT()
        
        signalInfo = Label(self,
                       text=a + "\n" + b + "\n" + c + "\n\n" + d + " \n" + e + "\n" + f  + "\n\n" + g,
                       background="white")
        signalInfo.place(x=80, y=50)

class DisplayDiffH(Toplevel):
    """
    Klasa tworzaca nowe okno, w ktorym otwierany jest canvas.

    """

    def __init__(self, master=None):

        super().__init__(master=master)
        self.title("Hamilton: " + fileNoDot)
        self.geometry("300x200")

        label = Label(self,
                      text="Informacje o analizie sygnalu",)
        label.pack()
        self.resizable(0, 0)
        
        a,b,c,d,e,f,g = AnalizeSignalH()
        
        signalInfo = Label(self,
                       text=a + "\n" + b + "\n" + c + "\n\n" + d + " \n" + e + "\n" + f  + "\n\n" + g,
                       background="white")
        signalInfo.place(x=80, y=50)

class DisplayDiffSWT(Toplevel):
    """
    Klasa tworzaca nowe okno, w ktorym otwierany jest canvas. 

    """

    def __init__(self, master=None):

        super().__init__(master=master)
        self.title("SWT: " + fileNoDot)
        self.geometry("300x200")

        label = Label(self,
                      text="Informacje o analizie sygnalu",)
        label.pack()
        self.resizable(0, 0)
        
        a,b,c,d,e,f,g = AnalizeSignalSWT()
        
        signalInfo = Label(self,
                       text=a + "\n" + b + "\n" + c + "\n\n" + d + " \n" + e + "\n" + f  + "\n\n" + g,
                       background="white")
        signalInfo.place(x=80, y=50)


def DummieCommand():
    """
    Kukla funkcji.

    Returns:
        True
    """
    return 1

# Canvas
canvas = Canvas(master,
                width=500,
                height=400,
                background="white")
canvas.place(x=150, y=100)

#  Napisy
title = Label(master,
              text='Program porownujacy metody\nanalizy sygnalu EKG',
              font='Bold 20',
              background="white")
title.place(x=220, y=130)

title = Label(master,
              text='Wybierz plik:',
              font='Bold 12',
              background="white")
title.place(x=200, y=225)

title = Label(master,
              text='Wykonaj: ',
              font='Bold 12',
              background="white")
title.place(x=205, y=275)

title = Label(master,
              text='Pan-Tompkins',
            #   font='1',
              background="white")
title.place(x=200, y=308)

title = Label(master,
              text='Hamilton',
            #   font='1',
              background="white")
title.place(x=210, y=338)

title = Label(master,
              text='SWT',
            #   font='1',
              background="white")
title.place(x=218, y=368)

# Przyciski
chooseFileButton = Button(master,
                          text='Wyszukaj plik',
                          width=14,
                          background="white",
                          command=ChooseFile
                          )
chooseFileButton.place(x=300, y=225)
tooltip1.bind(chooseFileButton,'Wybierz plik zawierajacy sygnal\nz plikow na komputerze ')

b1 = Button(master,
            text='Wyswietl sygnal',
            width=14,
            background="white",
            command=PlotSignal
            )
b1.place(x=300, y=275)
tooltip1.bind(b1,'Wyswietl sygnal wraz z anotacjami\nzamieszczonymi w pliku .ann przez lekarzy')

b2 = Button(master,
            text='Analiza tempa',
            width=14,
            background="white",
            command=Signal_and_bmp_plot
            )
b2.place(x=410, y=275)
tooltip1.bind(b2,'Wyswietl analize tempa metoda\nzaproponowana przez autora')

b3 = Button(master,
            text='Wyswietl sygnal',
            width=14,
            background="white",
            command=PanTompkins
            )
b3.place(x=300, y=305)

tooltip1.bind(b3,'Wyswietl sygnal wraz z wykrytymi zespolami\nQRS za pomoca algorytmu Pan-Tompkins')

b4 = Button(master,
            text='Porownaj z .ann',
            width=14,
            background="white",
            )
b4.place(x=410, y=305)
tooltip1.bind(b4,'Porownaj wykryte zespoly QRS z tymi\nzapisanymi w pliku .ann w bazie PhysioNet')
b4.bind("<Button>",
        lambda e: DisplayDiffPT(master))

b5 = Button(master,
            text='Wyswietl sygnal',
            width=14,
            background="white",
            command=Hamilton
            )
b5.place(x=300, y=335)
tooltip1.bind(b5,'Wyswietl sygnal wraz z wykrytymi zespolami\nQRS za pomoca algorytmu Hamilton')

b6 = Button(master,
            text='Porownaj z .ann',
            width=14,
            background="white",
            )
b6.place(x=410, y=335)
tooltip1.bind(b6,'Porownaj wykryte zespoly QRS z tymi\nzapisanymi w pliku .ann w bazie PhysioNet')
b6.bind("<Button>",
        lambda e: DisplayDiffH(master))

b7 = Button(master,
            text='Wyswietl sygnal',
            width=14,
            background="white",
            command=SWT
            )
b7.place(x=300, y=365)
tooltip1.bind(b7,'Wyswietl sygnal wraz z wykrytymi zespolami\nQRS za pomoca algorytmu SWT')

b8 = Button(master,
            text='Porownaj z .ann',
            width=14,
            background="white",
            )
b8.place(x=410, y=365)
tooltip1.bind(b8,'Porownaj wykryte zespoly QRS z tymi\nzapisanymi w pliku .ann w bazie PhysioNet')
b8.bind("<Button>",
        lambda e: DisplayDiffSWT(master))

b9 = Button(master,
            text='Eksportuj csv',
            width=14,
            background="white",
            command=ExportCSV_PT
            )
b9.place(x=520, y=305)
tooltip1.bind(b9,'Eksportuj plik z numerami probek\nwykrytych zespolow QRS')

b10 = Button(master,
            text='Eksportuj csv',
            width=14,
            background="white",
            command=ExportCSV_H
            )
b10.place(x=520, y=335)
tooltip1.bind(b10,'Eksportuj plik z numerami probek\nwykrytych zespolow QRS')

b11 = Button(master,
            text='Eksportuj csv',
            width=14,
            background="white",
            command=ExportCSV_SWT
            )
b11.place(x=520, y=365)
tooltip1.bind(b11,'Eksportuj plik z numerami probek\nwykrytych zespolow QRS')

mainloop()