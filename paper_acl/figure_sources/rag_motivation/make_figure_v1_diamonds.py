"""Rebuild the first-page RAG chart from audited local scores; no model calls.
Run with Python plus matplotlib and numpy. Outputs are next to this script.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT=Path(__file__).resolve().parent
rows=list(csv.DictReader((OUT/'rag_accuracy.csv').open()))
rag=np.array([100*float(r['rag_accuracy']) for r in rows])
fc=np.array([100*float(r['full_context_accuracy']) for r in rows])
gap=fc-rag
BLUE,RED,DARK='#4f81bd','#c0504d','#333333'
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],
 'mathtext.fontset':'stix','font.size':7.2,'axes.linewidth':.65,'xtick.major.width':.6,
 'ytick.major.width':.6,'xtick.major.size':0,'ytick.major.size':2.0,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(3.03,1.94))
fig.subplots_adjust(left=.145,right=.985,bottom=.33,top=.85)
x=np.arange(5)
ax.set_axisbelow(True)
ax.yaxis.grid(True,color='#e3e3e3',linewidth=.45)
ax.bar(x,rag,width=.59,color=[BLUE,BLUE,RED,RED,RED],edgecolor='black',linewidth=.45,zorder=2)
for i,(r,f,d) in enumerate(zip(rag,fc,gap)):
 ax.plot([i,i],[r,f],color=DARK,linewidth=.8,zorder=3)
 ax.text(i,r-12.0,f'{r:.1f}',ha='center',va='center',color='white',fontsize=7.1,fontweight='bold',zorder=5)
 ax.text(i,max(r,f)+4.0,f'{d:+.1f}',ha='center',va='bottom',color=DARK,fontsize=7.0,zorder=5)
ax.scatter(x,fc,marker='D',s=17,facecolor='white',edgecolor=DARK,linewidth=.8,zorder=4)
ax.set_ylim(0,112);ax.set_xlim(-.55,4.55)
ax.set_yticks([0,25,50,75,100]);ax.tick_params(axis='y',labelsize=6.8,pad=2)
ax.set_ylabel('Answer accuracy (%)',fontsize=7.5,labelpad=3)
names=['LoCoMo','LongMem.\nEval-S','MemEye','MemLens','VisualMem']
ax.set_xticks(x,[name+'\n$k='+str(row['retrieval_budget'])+'$' for name,row in zip(names,rows)],fontsize=6.9)
ax.tick_params(axis='x',pad=3)
ax.axvline(1.5,color='#a6a6a6',linewidth=.6,linestyle=(0,(2,2)))
ax.text(.5,118,'Text memory',ha='center',va='bottom',fontsize=7.6,color=BLUE,fontweight='bold',clip_on=False)
ax.text(3,118,'Multimodal memory',ha='center',va='bottom',fontsize=7.6,color=RED,fontweight='bold',clip_on=False)
handles=[Line2D([],[],marker='D',markersize=3.5,markerfacecolor='white',markeredgecolor=DARK,
 linestyle='',label='Full context')]
fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.18,.005),frameon=False,fontsize=6.8,
 handlelength=1.0,handletextpad=.4,borderaxespad=0)
fig.text(.985,.037,r'$\Delta = $ Full $-$ RAG (pp)',ha='right',va='center',fontsize=6.8,color=DARK)
for ext in ['pdf','svg','png']:
 fig.savefig(OUT/f'rag_motivation.{ext}',dpi=300,facecolor='white',bbox_inches='tight',pad_inches=.02)
plt.close(fig)
