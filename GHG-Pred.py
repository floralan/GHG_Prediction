"""
Programming date: Jan. 1st 2018
Author: Flora lan
Purpose: (1) to extract data from excel files, 
         (2) to build prediction models of four types of gases based on observed data
"""
import xlrd
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import itertools
##############################################################################
#                      Setting up for figure plot                            #
##############################################################################
def set_style():
    #pl.rc("font", family="Times New Roman")
    rc('mathtext', default='regular')
    #sns.set_style('ticks')
    sns.set(style="ticks", rc={'axes.linewidth': 1.5, 'font.family': "Times New Roman"})
    sns.set_context("paper",font_scale = 2.0, rc={"font.size":20,
                                                  "axes.titlesize":20,
                                                  "axes.labelsize":20,
                                                  "lines.linewidth": 2.0})
#############################################################################
#                     Reading Data from Excel file                          #
#############################################################################
def readExFile(fileName):
    nameList = []
    timeList = []
    variList = []
    workbook = xlrd.open_workbook(fileName)
    sheet0 = workbook.sheet_by_index(0)
    name, nrows, ncols = sheet0.name,sheet0.nrows, sheet0.ncols
    for i in range(0,nrows):
        if (i <= 1):
            pass
        elif (i == 2):
            temp = sheet0.row_values(i)[3:]
            nameList = temp[:]
        else:
            timeList.append(sheet0.row_values(i)[2])
            variList.append(sheet0.row_values(i)[3:])
        print("Reading Completes!")
    return nameList, timeList, variList
#############################################################################
#           Ploting out results, including raw data and fitted curves       #
#############################################################################
def plotFig(time, name, vari, fitVari = None, fitInfo = None):
    yFit = []
    nLoop = np.size(name,0)
    xVal = np.array(time)
    yVal = np.array(vari)
    # time, fs and title are all lists
    minTime = np.min(xVal)
    maxTime = np.max(xVal)
    fig,axes = plt.subplots(nrows = 2, ncols =3, figsize=(12,8), edgecolor = 'k')
    lineStyle = '-'
    cols = itertools.cycle(('b','g','c','m','k'))
    text = ['(a)','(b)','(c)','(d)','(e)']
    num = 0
    for i in range(0,2):
        for j in range(0,3):
            line1 = axes[i][j].scatter
            line2 = axes[i][j].plot()
            rCoeff = fitInfo[i]['determination']
            if (num == 5):
                axes[i][j].axis('off')
                break
            yTemp = yVal[:,num]
            minP = np.min(yTemp)
            maxP = np.max(yTemp)
            #axes.set_title('Excess pore pressure vs Time')
            axes[i][j].text(0.5,0.1,text[num], horizontalalignment = 'center',
                transform = axes[i][j].transAxes)
            axes[i][j].set_xscale('linear')
            axes[i][j].set_yscale('linear')
            axes[i][j].set_xlabel('Time')
            axes[i][j].set_ylabel(name[num])
            axes[i][j].grid(True, which = 'both')
            axes[i][j].set_ylim(minP,maxP)
            axes[i][j].set_xlim(minTime,maxTime)
            axes[i][j].tick_params( axis='x', # changes apply to the x-axesis
                direction='in',
                length=3,
                width=2,
                which='both', # both major and minor ticks are affected
                bottom='on', # ticks along the bottom edge are off
                top='on', # ticks along the top edge are off
                labelbottom='on') # labels along the bottom edge are off
           
            axes[i][j].tick_params( axis='y', # changes apply to the y-axesis
                direction='in',
                length=3,
                width=2,
                which='both', # both major and minor ticks are affected
                left ='on', # ticks along the bottom edge are off
                right ='on', # ticks along the top edge are off
                labelbottom='on') # labels along the bottom edge are off
            if (fitVari == None):
                axes[i][j].plot(time,yTemp,
                    linestyle= lineStyle,
                    #markeredgecolor= 'none',
                    #marker = marker.next(),
                    color = next(cols),
                    label = 'Raw data')
                saveName = 'Flora_Project_raw'
            else:
                textTemp = "$\mathit{R^{2}}$ = "+ str(np.around(rCoeff,4))
                if (num == 3 or num == 4):
                    pos = [0.45,0.55]
                else:
                    pos = [0.05,0.85]
                axes[i][j].text(pos[0],pos[1],textTemp, horizontalalignment = 'left',
                transform = axes[i][j].transAxes)

            line1 = axes[i][j].scatter(time,yTemp,
                marker = 'o',
                #markeredgecolor= 'none',
                #marker = marker.next(),
                edgecolors = next(cols),
                label = 'Raw data')
            line2 = axes[i][j].plot(time,fitVari[num],
                        linestyle= lineStyle,
                        #markeredgecolor= 'none',
                        #marker = marker.next(),
                        color = next(cols),
                        label = 'Fitted curve')
            #axes[i][j].legend(line1, line1.get_label, loc = "best", fancybox = True)
            #axes[i][j].legend(line2, line2.get_label, loc = "best", fancybox = True)
            saveName = 'Flora_Project_fit'

        num += 1

fig.tight_layout(pad = 0.5)
fig.savefig( saveName + '.pdf', format = 'pdf',dip = 100)

#############################################################################
#                    Using Least-squared technique to fit data              #
#############################################################################

def polyfit(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()
    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x) # or [p(z) for z in x]
    ybar = np.sum(y)/len(y) # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2) # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2) # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot
    results['value'] = yhat

    return results

def CurveFit(name,time,vari):
    vari = np.array(vari)
    # Group A: linear; Group B: non-linear
    fitResults = []
    fitVariList = []
    num = np.size(name,0)
    for i in range(0,num):
        if (i == 0 or i == 2):
            fitResults.append(polyfit(time,vari[:,i],1))
            fitVariList.append(fitResults[i]['value'])
        elif (i == 1):
            fitResults.append(polyfit(time,vari[:,i],4))
            fitVariList.append(fitResults[i]['value'])
        else:
            fitResults.append(polyfit(time,vari[:,i],3))
            fitVariList.append(fitResults[i]['value'])
    return fitResults,fitVariList

set_style()

#nameList, timeList, variList = readExFile("CHG concentration.xlsx")
fitResults,fitVariList = CurveFit(nameList, timeList, variList)
plotFig(timeList,nameList, variList,fitVariList,fitResults)