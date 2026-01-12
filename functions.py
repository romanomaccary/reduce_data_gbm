import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
from astropy.io import fits

# 1. Search for ANY version of the file (using the * wildcard)
# Replace 'bn120123456' with your specific burst ID
def find_files(search_pattern):
    #search_pattern = f'{fermi_id}/glg_tcat_all_{fermi_id}_v*.fit'
    found_files = glob.glob(search_pattern)

    # 2. Check if we found anything
    if found_files:
        # 3. Sort the list. 
        # Since "v01" comes alphabetically after "v00", the last item is the newest.
        found_files.sort()
        best_file = found_files[-1]
        
        print(f"Using the latest version: {best_file}")
        return best_file 
        # Now you can open 'best_file' to check the detectors
    else:
        #print("No file found!")
        return Exception("No file found!")
    

def plot_MVT_(MVT_value,T90_value,fermi_id):

    # get the rows of df_long corresponding to these candidates

    df_missing_uplims = pd.read_csv('/astrodata/romain/MVT_GBM_paper/missing_grbs_uplims.txt', sep='\s+', header=None, names=['fermi_id','t90','et90','t90_start','FWHM','sig_FWHM_m','sig_FWHM_p'])
    df_missing_uplims=df_missing_uplims[df_missing_uplims.t90>1.5*df_missing_uplims.et90]
    df_missing_uplims_short = df_missing_uplims[df_missing_uplims.t90 < 2]
    df_missing_uplims_long = df_missing_uplims[df_missing_uplims.t90 >= 2]

    df_missing_vals = pd.read_csv('/astrodata/romain/MVT_GBM_paper/missing_grbs_vals.txt', sep='\s+', header=None, names=['fermi_id','t90','et90','t90_start','FWHM','sig_FWHM_m','sig_FWHM_p'])
    df_missing_vals
    # remove bn190508987, it is not a real missing value
    df_missing_vals = df_missing_vals[df_missing_vals.fermi_id != 'bn190508987']
    # remove GRB111220A, it is not a real missing value
    df_missing_vals = df_missing_vals[df_missing_vals.fermi_id != 'bn111220486']
    df_grb_web = pd.read_csv('/astrodata/romain/MVT_GBM_paper/GRB_web_name.txt', sep='\s+',names=['GRB','fermi_id'])
    df_grb_web

    df_merge = pd.merge(df_missing_vals, df_grb_web, on='fermi_id')
    df_merge
    # reorganize the columns to have GRB in second position
    df_merge = df_merge[['fermi_id', 'GRB', 't90', 'et90', 't90_start', 'FWHM', 'sig_FWHM_m', 'sig_FWHM_p']]
    df_merge

    path_fwhm_file = '/astrodata/romain/MVT_GBM_paper/FWHM_min_Fermi_all_GRBs_rem_wt90_wGRBname.txt'
    df2 = pd.read_csv(path_fwhm_file,sep='\s+',names=['fermi_id','GRB','t90','et90','t90_start','FWHM','sig_FWHM_m','sig_FWHM_p'])
    df = pd.concat([df_merge, df2], ignore_index=True)

    #df=df[df.t90>1*df.et90]
    SNGRBs =np.loadtxt('/astrodata/romain/MVT_GBM_paper/SNGRBs.txt',dtype=str)
    SNGRBs = ['GRB' + s for s in SNGRBs]
    df_long = df[df.t90>2]
    df_long = df_long[df_long.t90>1.5*df_long.et90]
    df_short = df[(df.t90<2)]
    df_short = df_short[df_short.t90>1.5*df_short.et90]


    magnetar_GFs = ['GRB180128A','GRB200415A','GRB231115A']
    long_mergers=['GRB191019A','GRB211211A','GRB230307A']
    kaneko_grbs = np.loadtxt('/astrodata/romain/MVT_GBM_paper/KanekoSEEGRBs.txt',dtype=str)
    # exclude 090927A from kaneko_grbs
    kaneko_grbs = kaneko_grbs[kaneko_grbs != 'bn090927422']
                            
    lan_grbs = np.loadtxt('/astrodata/romain/MVT_GBM_paper/LanSEEGRBs.txt',dtype=str)
    lien_grbs = np.loadtxt('/astrodata/romain/MVT_GBM_paper/LienSEEGRBs.txt',dtype=str)

    df_kaneko = df[df['fermi_id'].isin(kaneko_grbs)]
    df_kaneko = df_kaneko[df_kaneko.t90>1.2*df_kaneko.et90]
    df_lan    = df[df['fermi_id'].isin(lan_grbs)]
    df_lan    = df_lan[df_lan.t90>1.2*df_lan.et90]
    df_lien   = df[df['fermi_id'].isin(lien_grbs)]
    df_lien   = df_lien[df_lien.t90>1.2*df_lien.et90]
    # concatenate all SEE GRBs from lan kaneko and lien
    df_SEE = pd.concat([df_kaneko, df_lan, df_lien])

    df_magnetars = df[df['GRB'].isin(magnetar_GFs)]
    df_long_mergers = df[df['GRB'].isin(long_mergers)]
    df_SNGRB = df[df['GRB'].isin(SNGRBs)]

    fig = plt.figure(figsize=(10,10))
    plt.rcParams.update({'font.size': 18})

    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                        wspace=0.095, hspace=0.05)

    # Define axes
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)



    bins=np.logspace(-3,2,22)
    ax_histx.hist(df_long.FWHM, bins=bins,color='red',alpha=0.25,edgecolor='white',lw=0.3)
    ax_histx.hist(df_short.FWHM, bins=bins,color='blue',alpha=0.25,edgecolor='white',lw=0.3)



    ax_histx.hist(df_SEE.FWHM, bins=bins,color='grey',alpha=1,edgecolor='white',lw=0.3)
    ax_histx.hist(df_magnetars.FWHM, bins=bins,color='brown',alpha=1,edgecolor='white',lw=0.3)
    # ax_histx.hist(df_kaneko.FWHM, bins=bins,color='cyan',alpha=0.5)
    # #add lan_grbs
    # ax_histx.hist(df_lan.FWHM, bins=bins,color='lime',alpha=0.5)
    # #add lien_grbs
    # ax_histx.hist(df_lien.FWHM, bins=bins,color='darkblue',alpha=0.5)
    ax_histx.hist(df_SNGRB.FWHM, bins=bins,color='gold',alpha=1,edgecolor='white',lw=0.3)


    bins_t90=np.logspace(-1.5,3,20)

    ax_histy.hist(df_long.t90, bins=np.logspace(np.log10(np.min(df_long.t90)),np.log10(np.max(df_long.t90)),12),color='red',alpha=0.25, orientation='horizontal',edgecolor='white',lw=0.3)
    ax_histy.hist(df_short.t90, bins=np.logspace(np.log10(np.min(df_short.t90)),np.log10(np.min(df_long.t90)),12),color='blue',alpha=0.25, orientation='horizontal',edgecolor='white',lw=0.3)
    ax_histy.hist(df_SEE.t90, bins=bins_t90,color='grey',alpha=1, orientation='horizontal',edgecolor='white',lw=0.3)
    ax_histy.hist(df_SNGRB.t90, bins=bins_t90,color='gold',alpha=1, orientation='horizontal',edgecolor='white',lw=0.3)
    ax_histy.hist(df_magnetars.t90, bins=bins_t90,color='brown',alpha=1, orientation='horizontal',edgecolor='white',lw=0.3)
    # #add lan_grbs
    # ax_histy.hist(df_lan.t90, bins=np.logspace(np.log10(np.min(df_lan.t90)),np.log10(np.max(df_lan.t90)),10),color='lime',alpha=0.5, orientation='horizontal')
    # #add lien_grbs
    # ax_histy.hist(df_lien.t90, bins=np.logspace(np.log10(np.min(df_lien.t90)),np.log10(np.max(df_lien.t90)),10),color='darkblue',alpha=0.5, orientation='horizontal')

    ax_histx.set_xscale('log')
    ax_histx.set_yscale('log')

    ax_histy.set_xscale('log')
    ax_histy.set_yscale('log')


    #ax_histy.hist(y, bins=30, orientation='horizontal', color='gray')


    ax_scatter.scatter(df_long_mergers.FWHM,df_long_mergers.t90,c='k',s=200,marker='*')

    fwhm_1910,t90_1910 = 0.150,64
    ax_scatter.scatter(fwhm_1910,t90_1910,c='k',s=200,marker='*')
    # annotate 
    ax_scatter.text(1.05*fwhm_1910,1.05*t90_1910,'191019A',color='k',size=14)

    ks_x,ks_y=[0.5,0.75],[0.6,1.2]
    for i in range(2):

        wrt = df_long_mergers.GRB.values[i]
        ax_scatter.text(ks_x[i]*df_long_mergers.FWHM.values[i],ks_y[i]*df_long_mergers.t90.values[i],wrt[3:],color='k',size=14)

    ax_scatter.errorbar(df_long.FWHM,df_long.t90,xerr=[-df_long.sig_FWHM_m,df_long.sig_FWHM_p],yerr=df_long.et90,ls='',c='r',alpha=0.3)
    ax_scatter.errorbar(df_short.FWHM,df_short.t90,xerr=[-df_short.sig_FWHM_m,df_short.sig_FWHM_p],yerr=df_short.et90,ls='',c='b',alpha=0.4)


    ax_scatter.plot(np.array([1e-3,1e3]),np.array([1e-3,1e3]),c='k',lw=3)
    ax_scatter.plot(np.array([1e-3,1e3]),10*np.array([1e-3,1e3]),c='k',lw=1.5,ls='--')
    ax_scatter.plot(np.array([1e-3,1e3]),100*np.array([1e-3,1e3]),c='k',lw=1.5,ls='dashdot')
    ax_scatter.plot(np.array([1e-3,1e3]),1000*np.array([1e-3,1e3]),c='k',lw=1.5,ls='dotted')


    #ax_scatter.errorbar(df_magnetars.FWHM.values[0],df_magnetars.t90.values[0],uplims=True,xerr=[-df_magnetars.sig_FWHM_m.values[0],df_magnetars.sig_FWHM_p.values[0]],yerr=df_magnetars.et90.values[0],ls='',c='brown',lw=3)

    # I want to plot only the last two magnetars with upper limits and the first one with error bars
    # but I want to keep the error bars for the first one
    # so I will loop through the magnetars and plot them one by one
    # and I will plot the first one with error bars and the others with upper limits
    for i in range(3):
        if i == 1:
            ax_scatter.errorbar(
                df_magnetars.FWHM.values[i],
                df_magnetars.t90.values[i],
                xerr=[[-df_magnetars.sig_FWHM_m.values[i]], [df_magnetars.sig_FWHM_p.values[i]]],
                yerr=df_magnetars.et90.values[i],
                ls='', c='brown', lw=3
            )
        else:
            ax_scatter.errorbar(
                df_magnetars.FWHM.values[i],
                df_magnetars.t90.values[i],
                xerr=[[-df_magnetars.sig_FWHM_m.values[i]], [df_magnetars.sig_FWHM_p.values[i]]],
                yerr=df_magnetars.et90.values[i]/5,
                c='brown', lw=3, uplims=True,  # <-- use this if t90 is an upper limit
            )
    ax_scatter.errorbar(df_kaneko.FWHM,df_kaneko.t90,xerr=[-df_kaneko.sig_FWHM_m,df_kaneko.sig_FWHM_p],yerr=df_kaneko.et90,ls='',c='cyan',lw=3)
    ax_scatter.errorbar(df_lan.FWHM,df_lan.t90,xerr=[-df_lan.sig_FWHM_m,df_lan.sig_FWHM_p],yerr=df_lan.et90,ls='',c='lime',lw=3)
    ax_scatter.errorbar(df_lien.FWHM,df_lien.t90,xerr=[-df_lien.sig_FWHM_m,df_lien.sig_FWHM_p],yerr=df_lien.et90,ls='',c='magenta',lw=3,alpha=1)
                

    ax_scatter.errorbar(df_missing_uplims_long.FWHM.values,df_missing_uplims_long.t90.values,xerr=[-df_missing_uplims_long.sig_FWHM_m.values, df_missing_uplims_long.sig_FWHM_p.values],yerr=df_missing_uplims_long.et90.values,c='red',ls='', lw=1., xuplims=True)  # <-- use this if t90 is an upper limit
    ax_scatter.errorbar(df_missing_uplims_short.FWHM.values,df_missing_uplims_short.t90.values,xerr=[-df_missing_uplims_short.sig_FWHM_m.values, df_missing_uplims_short.sig_FWHM_p.values],yerr=df_missing_uplims_short.et90.values,c='blue',ls='', lw=1., xuplims=True)  # <-- use this if t90 is an upper limit

    ax_scatter.errorbar(df_SNGRB.FWHM,df_SNGRB.t90,xerr=[-df_SNGRB.sig_FWHM_m,df_SNGRB.sig_FWHM_p],yerr=df_SNGRB.et90,ls='',c='gold',lw=2)
    ax_scatter.plot(df_SNGRB.FWHM.values,df_SNGRB.t90.values,'s',c='gold')
    # find a flashy color for the candidates

    #for j in range(len(df_candidates)):
    #    ax_scatter.plot(df_candidates.FWHM.values[j],df_candidates.t90.values[j],'o',c=df_candidates.color.values[j],ms=8,mec='k',mew=1,label='GRB '+df_candidates.GRB_name.values[j])
    #ax_scatter.plot(df_candidates.FWHM,df_candidates.t90,'X',c='grey',ms=12)
    #ax_scatter.errorbar(df_candidates.FWHM,df_candidates.t90,xerr=[-df_candidates.sig_FWHM_m,df_candidates.sig_FWHM_p],yerr=df_candidates.et90,ls='',c='pink',lw=2)

    ks_x,ks_y = [0.5,0.5,1.2],[1.2,0.5,0.6]
    for i in range(3):
        wrt=df_magnetars.GRB.values[i]
        ax_scatter.text(ks_x[i]*df_magnetars.FWHM.values[i],ks_y[i]*df_magnetars.t90.values[i],wrt[3:],color='brown',size=14)




    ax_scatter.set_xlabel(r'${\rm FWHM_{min}}$'+' [s]',size=24)
    ax_scatter.set_ylabel(r'${\rm T_{90}}$'+' [s]',size=24)

    ax_scatter.loglog()
    ax_scatter.set_xlim(1e-3,100)
    ax_scatter.set_ylim(1e-2,1000)
    plt.setp(ax_histx.get_xticklabels(), visible=False)
    plt.setp(ax_histy.get_yticklabels(), visible=False)
    fig.tight_layout()

    ax_scatter.annotate('200826A',
                xy=(0.1, 1),             # Point to annotate
                xytext=(10, 0.1),       # Position of the annotation text
                arrowprops=dict(arrowstyle='->', color='gold', lw=2.5),
                fontsize=14,
                color='k')

    fig.legend(fontsize=8,bbox_to_anchor=(0.25,0.7))

    ax_scatter.plot([MVT_value], [T90_value], marker='s', markersize=16, color='green',label=f'{ fermi_id}, MVT={MVT_value} s')
    ax_scatter
#    plt.show()
    ax_scatter.legend(fontsize=14, loc='upper left')

def log_likelihood(theta, x, y, xerr, yerr):
    m, q, sigma = theta
    model = m*x + q
    denominator = sigma**2 + yerr**2 + m**2*xerr**2
    return -0.5*np.sum(np.log(1/(2*np.pi*denominator)) - (y - model)**2/denominator)

def plot_Amati_rel():

    E_iso_fit = np.logspace(47, 55, 100)
    K_L = 200
    m_L = 0.36
    sigma_L = 0.28
    E_p_fit = K_L * (E_iso_fit / 1e52)**m_L

    E_p_m = E_p_fit * 10**(-sigma_L)
    E_p_p = E_p_fit * 10**(sigma_L)

    # for 2 sigma_L
    E_p_m_2sigma = E_p_fit * 10**(-2*sigma_L)
    E_p_p_2sigma = E_p_fit * 10**(2*sigma_L)

    # for 3 sigma_L

    E_p_m_3sigma = E_p_fit * 10**(-3*sigma_L)
    E_p_p_3sigma = E_p_fit * 10**(3*sigma_L)

    K_s = 1244
    m_s = 0.361
    sigma_s = 0.38
    E_p_fit_s = K_s * (E_iso_fit / 1e52)**m_s


    E_p_m_s_1sigma = E_p_fit_s * 10**(-sigma_s)
    E_p_p_s_1sigma = E_p_fit_s * 10**(sigma_s)

    # for 2-sigma
    E_p_m_s_2sigma = E_p_fit_s * 10**(-2*sigma_s)
    E_p_p_s_2sigma = E_p_fit_s * 10**(2*sigma_s)

    redshift_dataframe = pd.read_csv('/astrodata/romain/follow_up_230307A/info_grb_files/alles_GRBs_mit_z_without_ref.txt',delim_whitespace=True,comment='#',header=None,names=['GRB','z','t90','class','ref'])
    data_frame_liso1 = pd.read_csv('/astrodata/romain/follow_up_230307A/info_grb_files/logEpi_logEiso_logLiso.dat',delim_whitespace=True,comment='#',header=None,names=['GRB','logEp','log_Ep_err','logEiso','logEiso_err','logLiso','logLiso_err','Ref'])
    data_frame_liso = pd.merge(data_frame_liso1, redshift_dataframe, on='GRB')
    data_frame_liso

    data_frame_lisoL = data_frame_liso[data_frame_liso['class']=='L']
    data_frame_lisoS = pd.read_csv('/astrodata/romain/KW_spec/log_Epi_log_Eiso_log_Liso_short.txt',delim_whitespace=True,comment='#',header=None,names=['GRB','logEp','log_Ep_err','logEiso','logEiso_err','logLiso','logLiso_err','Ref'],skiprows=1)

    data_frame_lisoL = data_frame_lisoL[data_frame_lisoL.logEiso>5]
    data_frame_lisoS = data_frame_lisoS[data_frame_lisoS.logEiso>5]

    # exclude points with large error bars
    data_frame_lisoL = data_frame_lisoL[(data_frame_lisoL.log_Ep_err<1*data_frame_lisoL.logEp) & (data_frame_lisoL.logEiso_err<1*data_frame_lisoL.logEiso)]
    logEp_obsL,log_Ep_err_obsL,logEiso_obsL,logEiso_err_obsL  = data_frame_lisoL['logEp'].values,data_frame_lisoL['log_Ep_err'].values,data_frame_lisoL['logEiso'].values,data_frame_lisoL['logEiso_err'].values

    # idem for short GRBs
    data_frame_lisoS = data_frame_lisoS[(data_frame_lisoS.log_Ep_err<1*data_frame_lisoS.logEp) & (data_frame_lisoS.logEiso_err<1*data_frame_lisoS.logEiso)]
    logEp_obsS,log_Ep_err_obsS,logEiso_obsS,logEiso_err_obsS  = data_frame_lisoS['logEp'].values,data_frame_lisoS['log_Ep_err'].values,data_frame_lisoS['logEiso'].values,data_frame_lisoS['logEiso_err'].values


    Ep_obsL = 10**logEp_obsL
    Eiso_obsL = 10**logEiso_obsL

    Ep_obsS = 10**logEp_obsS
    Eiso_obsS = 10**logEiso_obsS


    plt.figure(figsize=(14, 10))

    plt.xlim(1e47,1e55)
    plt.ylim(1e1,1e4)


    plt.plot(logEp_obsL,logEiso_obsL, 'o', color='maroon', label='Type II GRBs data', alpha=0.5)
    plt.errorbar(Eiso_obsL, Ep_obsL, xerr=Eiso_obsL * logEiso_err_obsL, yerr=Ep_obsL * log_Ep_err_obsL, fmt='o', color='maroon', alpha=0.5, capsize=3)


    plt.plot(Eiso_obsS,Ep_obsS, 'o', color='darkblue', label='Type I GRBs data', alpha=0.5)
    plt.errorbar(Eiso_obsS, Ep_obsS, xerr=Eiso_obsS * logEiso_err_obsS, yerr=Ep_obsS * log_Ep_err_obsS, fmt='o', color='darkblue', alpha=0.5, capsize=3)

    plt.fill_between(E_iso_fit, E_p_m, E_p_p, color='red', alpha=0.3,label='1-sigma region')
    # for 2-sigma   
    plt.fill_between(E_iso_fit, E_p_m_2sigma, E_p_p_2sigma, color='red', alpha=0.2,label='2-sigma region')
    # for 3-sigma
    plt.fill_between(E_iso_fit, E_p_m_3sigma, E_p_p_3sigma, color='red', alpha=0.1,label='3-sigma region')

    plt.loglog(E_iso_fit, E_p_fit, 'r-', lw=2,label='Amati relation')

    plt.xlabel(r'$E_{\rm iso}$ [erg]', fontsize=30)
    plt.ylabel(r'$E_{\rm p,i}$ [keV]', fontsize=30)
    plt.grid(True, which="both", ls=":")
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

