# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 20:27:09 2025

@author: bcamc
"""
#%% Import packages

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean
import scipy

#%% FIG 1: Line P + MVP Longitude map + boxplot - model vs smoothed obs. + lit algos

fig = plt.figure(figsize=(24,12), dpi=300)
font={'family':'sans-serif',
      'weight':'normal',
      'size':'24'}
plt.rc('font', **font) # sets the specified font formatting globally
gs = fig.add_gridspec(2,5)
ax = fig.add_subplot(gs[0,2:5])
ax2 = fig.add_subplot(gs[1,2:5])
ax3 = fig.add_subplot(gs[0:2,0:2])

#==============================================================================

boxdata = [
    PMEL[PMEL['DateTime'].dt.month == 8]['swDMS'],
    DMS_ship_anom.iloc[:DMS_coords['P26']]['DMS'].dropna(),
    DMS_ship_anom.iloc[DMS_coords['P20']:DMS_coords['P26']]['DMS'].dropna(),
    ]
labels = ['Climatology \n(1997-2017)',
          'Line P \n(2022)',
          'P20-P26 \n(2022)']
bplot1 = ax3.boxplot(boxdata,
                    widths=0.5,
                    vert=False,
                    patch_artist=True,  # fill with color
                    labels=labels,
                    )

# fill with colors
colors = ['gray', 'red', 'red']
for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
for patch in bplot1['medians']:
    patch.set_color('black')

ax3.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
ax3.tick_params('both', length=10, width=1, which='major')
ax3.tick_params('both', length=5, width=1, which='minor')

ax3.set_xlabel('DMS (nM)')

ax3.text(0.03,0.95, 'a', color='k', fontweight='bold', fontsize=24,
         transform=ax3.transAxes)

#==============================================================================
# Line P
ax.plot(DMS_ship_anom.iloc[:DMS_coords['P26'],:].index.get_level_values('lons'),
        DMS_ship_anom.iloc[:DMS_coords['P26']]['DMS'],
        ls='None',
        marker='.',
        c='k',
        ms=10,
        zorder=3,
        label='Aug 2022 Obs.')

result = lowess(DMS_ship_anom.iloc[:DMS_coords['P26']]['DMS'].values,
                DMS_ship_anom.iloc[:DMS_coords['P26'],:].index.get_level_values('lons').values,
                frac=0.2)
x_smooth = result[:,0]
y_smooth = result[:,1]
ax.plot(x_smooth,
        y_smooth,
        color='k',
        ls='--',
        lw=3,
        zorder=3,
        label='LOWESS smoothing')

ax_2 = ax.twinx()

ax_2.plot(DMSP_ship_anom.iloc[:DMS_coords['P26'],:].index.get_level_values('lons'),
        DMSP_ship_anom.iloc[:DMS_coords['P26']]['DMSP'],
        ls='None',
        marker='.',
        c='r',
        alpha=0.5,
        ms=10,
        zorder=2,
        label='Aug 2022 Obs.')

result = lowess(DMSP_ship_anom.iloc[:DMS_coords['P26']]['DMSP'].values,
                DMSP_ship_anom.iloc[:DMS_coords['P26'],:].index.get_level_values('lons').values,
                frac=0.2)
x_smooth = result[:,0]
y_smooth = result[:,1]
ax_2.plot(x_smooth,
        y_smooth,
        color='r',
        ls='--',
        lw=3,
        zorder=2,
        label='LOWESS smoothing')

ax_2.tick_params(axis='y', color='r', labelcolor='r')
ax_2.set_ylabel('DMSP (nM)', color='r')

# ax.set_zorder(ax2.get_zorder()+1)
ax.set_ylabel('DMS (nM)')
ax.set_xticklabels([])

ax.text(0.03,0.9, 'b', color='k', fontweight='bold', fontsize=24,
         transform=ax.transAxes)

#------------------------------------------------------------------------------
# Plot baseline model

ax.plot(models_combined.iloc[model_inds1].index.get_level_values('lonbins'),
        models_combined.iloc[model_inds1],
        color='b',
        # marker='.',
        ls='-',
        lw=4,
        ms=20,
        label='Climatology (RFR+ANN)')

ax.plot(models_combined.iloc[model_inds1].index.get_level_values('lonbins'),
        models_combined.iloc[model_inds1].values.flatten() + models_sd.iloc[model_inds1].values.flatten(),
        ls='--',
        lw=2,
        color='b',
        alpha=0.2)
ax.plot(models_combined.iloc[model_inds1].index.get_level_values('lonbins'),
        models_combined.iloc[model_inds1].values.flatten() - models_sd.iloc[model_inds1].values.flatten(),
        ls='--',
        lw=2,
        color='b',
        alpha=0.2)
ax.fill_between(models_combined.iloc[model_inds1].index.get_level_values('lonbins'),
                models_combined.iloc[model_inds1].values.flatten() + models_sd.iloc[model_inds1].values.flatten(),
                models_combined.iloc[model_inds1].values.flatten() - models_sd.iloc[model_inds1].values.flatten(),
                color='b',
                alpha=0.2)

#------------------------------------------------------------------------------

lineP_ax = ax.secondary_xaxis('top')
lineP_ax.set_xticks(SSN.loc[['P4','P12','P16','P20','P26']].index.get_level_values('lon'))
lineP_ax.set_xticklabels(SSN.loc[['P4','P12','P16','P20','P26']].index.get_level_values('station'), fontdict={'fontsize':24})

for x in lineP_ax.get_xticks():
    ax.axvline(x, color='gray',
                zorder=-1,
                linestyle="-.",
                linewidth=1,
                dashes=(5, 5),
                )
ax.set_xlim(-145.5, -124.3)
ax.tick_params('both', length=7)
ax.set_ylim(0,68)


axins = inset_axes(ax, width="40%", height="40%", loc="upper center", 
                    bbox_to_anchor=(0.25,0.5,1,0.5),
                    bbox_transform=ax.transAxes,
                axes_class=GeoAxes, 
                axes_kwargs=dict(projection=ccrs.PlateCarree())
                )

axins.scatter(uw_DMS.iloc[:DMS_OSP_loc,:].loc[:,'lon'],
              uw_DMS.iloc[:DMS_OSP_loc,:].loc[:,'lat'],
              s=10,
              c='k',
              transform=ccrs.PlateCarree())

for stn in LineP_stns.index[1:]:
    axins.text(x=LineP_stns['lon'].loc[stn],
            y=LineP_stns['lat'].loc[stn]-0.3,
            s=stn,
            ha='center',
            va='top',
            fontsize=12,
            transform=ccrs.PlateCarree())

axins.add_feature(cartopy.feature.LAND, edgecolor='None', facecolor='darkgray', zorder=2)
axins.add_feature(cartopy.feature.COASTLINE, edgecolor='k', zorder=2)
axins.set_extent([-146, -123, 47.5, 52])

#==============================================================================
# MVP line
#------------------------------------------------------------------------------
ax2.plot(DMS_ship_anom.iloc[DMS_coords['P26']:,:].index.get_level_values('lons'),
        DMS_ship_anom.iloc[DMS_coords['P26']:]['DMS'],
        ls='None',
        marker='.',
        c='k',
        ms=10,
        zorder=2,
        label='Aug 2022 Obs.')

result = lowess(DMS_ship_anom.iloc[DMS_coords['P26']:]['DMS'].values,
                DMS_ship_anom.iloc[DMS_coords['P26']:,:].index.get_level_values('lons').values,
                frac=0.2)
x_smooth = result[:,0]
y_smooth = result[:,1]
ax2.plot(x_smooth,
        y_smooth,
        color='k',
        ls='--',
        lw=3,
        label='LOWESS smoothing')

ax_3 = ax2.twinx()

ax_3.plot(DMSP_ship_anom.iloc[DMS_coords['P26']:,:].index.get_level_values('lons'),
        DMSP_ship_anom.iloc[DMS_coords['P26']:]['DMSP'],
        ls='None',
        marker='.',
        c='r',
        alpha=0.5,
        ms=10,
        zorder=2,
        label='Aug 2022 Obs.')

result = lowess(DMSP_ship_anom.iloc[DMS_coords['P26']:]['DMSP'].values,
                DMSP_ship_anom.iloc[DMS_coords['P26']:,:].index.get_level_values('lons').values,
                frac=0.2)
x_smooth = result[:,0]
y_smooth = result[:,1]
ax_3.plot(x_smooth,
        y_smooth,
        color='r',
        ls='--',
        lw=3,
        label='LOWESS smoothing')

ax_3.tick_params(axis='y', color='r', labelcolor='r')
ax_3.set_ylabel('DMSP (nM)', color='r')

ax2.set_ylabel('DMS (nM)')
label_coords = np.arange(-144, -126+3, 3)
ax2.set_xticks(label_coords)
# format the labels into coordinates
ax2.set_xticklabels([str(abs(i))+'$^{\mathrm{o}}$W' for i in label_coords], rotation=360)
ax2.set_xlabel('Longitude ($^{o}$W)')

ax2.text(0.03,0.9, 'c', color='k', fontweight='bold', fontsize=24,
         transform=ax2.transAxes)

#------------------------------------------------------------------------------
# Plot baseline model

ax2.plot(models_combined.iloc[model_inds2].index.get_level_values('lonbins'),
        models_combined.iloc[model_inds2],
        color='b',
        ls='-',
        lw=4,
        ms=20,
        label='Climatology (RFR+ANN)')

ax2.plot(models_combined.iloc[model_inds2].index.get_level_values('lonbins'),
        models_combined.iloc[model_inds2].values.flatten() + models_sd.iloc[model_inds2].values.flatten(),
        ls='--',
        lw=2,
        color='b',
        alpha=0.2)
ax2.plot(models_combined.iloc[model_inds2].index.get_level_values('lonbins'),
        models_combined.iloc[model_inds2].values.flatten() - models_sd.iloc[model_inds2].values.flatten(),
        ls='--',
        lw=2,
        color='b',
        alpha=0.2)
ax2.fill_between(models_combined.iloc[model_inds2].index.get_level_values('lonbins'),
                models_combined.iloc[model_inds2].values.flatten() + models_sd.iloc[model_inds2].values.flatten(),
                models_combined.iloc[model_inds2].values.flatten() - models_sd.iloc[model_inds2].values.flatten(),
                color='b',
                alpha=0.2,
                label='$\pm$1 SD (RFR + ANN)')

ax2.legend(fontsize=20, loc='center right',)
#------------------------------------------------------------------------------

ax2.set_xlim(-145.5, -124.3)

ax2.tick_params('both', length=7)
ax2.set_ylim(0,68)

axins = inset_axes(ax2, width="40%", height="40%", loc="upper center", 
                    bbox_to_anchor=(0.25,0.5,1,0.5),
                    bbox_transform=ax2.transAxes,
                    axes_class=GeoAxes, 
                    axes_kwargs=dict(projection=ccrs.PlateCarree())
                    )

axins.scatter(uw_DMS.iloc[210:,:].loc[:,'lon'],
              uw_DMS.iloc[210:,:].loc[:,'lat'],
              s=10,
              c='k',
              transform=ccrs.PlateCarree())

axins.add_feature(cartopy.feature.LAND, edgecolor='None', facecolor='darkgray', zorder=2)
axins.add_feature(cartopy.feature.COASTLINE, edgecolor='k', zorder=2)
axins.set_extent([-146, -123, 47.5, 52])

fig.subplots_adjust(wspace=0.8)

fig.savefig(fig_dir+'Fig_1.tif', bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})

#%% FIG 2: map SSTA, DMS boxplots and 5-d averaged SSTA vs DMS:chl

# MHW_chl_matched = MHW_matched.loc[matched_8d_nona['datetime']].copy()
# MHW_chl_matched['DMS:chl'] = pd.Series(MHW_chl_matched['DMS'].values / matched_8d_nona['chl'].values, index=MHW_chl_matched.index)

fig = plt.figure(figsize=(24,12), dpi=300)
gs = fig.add_gridspec(2, 2)
ax = fig.add_subplot(gs[0,:], projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(gs[1,1])
ax3 = fig.add_subplot(gs[1,0])

ax_1 = ax3.twinx()

#------------------------------------------------------------------------------
ds = [str(i).zfill(2) for i in np.arange(10,24+1,1)]
SSTA_plot = SST_anom_stack.loc[ds].stack().groupby(['lat','lon']).mean().unstack('lon')
SSTA_plot = SSTA_plot.loc[sst_mean.index, sst_mean.columns]

# for visualizing heatwave bounds: calculate ~90th percentile from SD and z-scores
diff = (1.28*sst_std)
cat1_mask = SSTA_plot[((SSTA_plot >= diff) & (SSTA_plot < 2*diff))]
cat2_mask = SSTA_plot[((SSTA_plot >= 2*diff) & (SSTA_plot < 3*diff))]
cat3_mask = SSTA_plot[((SSTA_plot >= 3*diff) & (SSTA_plot < 4*diff))]

cat1_mask = cat1_mask.where(pd.notna(cat1_mask), 0).where(pd.isna(cat1_mask), 1)
cat2_mask = cat2_mask.where(pd.notna(cat2_mask), 0).where(pd.isna(cat2_mask), 1)
cat3_mask = cat3_mask.where(pd.notna(cat3_mask), 0).where(pd.isna(cat3_mask), 1)

h = ax.pcolormesh(SSTA_plot.columns,
            SSTA_plot.index,
            SSTA_plot.values,
            vmin=1,
            vmax=2.5,
            cmap=mpl.cm.Reds,
            transform=ccrs.PlateCarree())


ax.contour(cat1_mask.columns,
            cat1_mask.index,
            cat1_mask.values,
            levels=[0],
            colors='k',
            transform=ccrs.PlateCarree())

h2 = ax.scatter(uw_DMS_ind.loc[:,'lon'],
           uw_DMS_ind.loc[:,'lat'],
           s=30,
           vmin=0,
           vmax=30,
           c=uw_DMS_ind.loc[:,'conc'],
           zorder=2,
           transform=ccrs.PlateCarree())

for stn in LineP_stns.index[1:]:
    ax.text(x=LineP_stns['lon'].loc[stn],
            y=LineP_stns['lat'].loc[stn]-0.1,
            s=stn,
            ha='center',
            va='top',
            transform=ccrs.PlateCarree())

gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
ax.add_feature(cartopy.feature.LAND, edgecolor='None', facecolor='darkgray', zorder=2)
ax.add_feature(cartopy.feature.COASTLINE, edgecolor='k', zorder=2)
ax.set_extent([-146, -123, 48.25, 52])

ax_cb = fig.add_axes([0.1, 0.88, 0.6, 0.02])
fig.add_axes(ax_cb)
cb1 = plt.colorbar(h, cax=ax_cb, orientation='horizontal', extend='max', extendfrac=0.02)
cb1.ax.tick_params(labelsize=20)
ax_cb.xaxis.set_ticks_position('top')
ax_cb.xaxis.set_label_position('top')
cb1.set_label('SST anomaly ($^{o}$C)', fontsize=20, labelpad=10)
cb1.ax.invert_xaxis()

divider = make_axes_locatable(ax)
ax_cb = divider.new_horizontal(size="2%", pad=0.2, axes_class=plt.Axes)
fig.add_axes(ax_cb)
cb1 = plt.colorbar(h2, cax=ax_cb, extend='max')
cb1.ax.tick_params(labelsize=20)
ax_cb.xaxis.set_ticks_position('top')
ax_cb.xaxis.set_label_position('top')
cb1.set_label('DMS (nM)', fontsize=20, labelpad=10)

ax.text(0.01,0.85, 'a', color='k', fontweight='bold', fontsize=24,
         transform=ax.transAxes)
#------------------------------------------------------------------------------

ax2.scatter(SSTA_mat.mean(axis=1).loc[matched_8d_nona['datetime']],
           matched_8d_nona['DMS']/matched_8d_nona['chl'],
           c='k',
           s=30)

ax2.set_ylabel('DMS:chl-a (nmol $\mu$g$^{-1}$)')
ax2.set_xlabel('5-day mean SST anomaly ($^{o}$C)')

ax2.text(0.03,0.9, 'c', color='k', fontweight='bold', fontsize=24,
         transform=ax2.transAxes)

#------------------------------------------------------------------------------

space = 0.15
groups = MHW_chl_matched.loc[:,'MHW'].unique()
box_param = dict(
    # whis=(5, 95),
    widths=0.2,
    patch_artist=True,
    boxprops=dict(ec='k'),
    capprops=dict(color='k', lw=1),
    whiskerprops=dict(color='k', lw=1),
    flierprops=dict(color='k', marker='.', mec='k', fillstyle=None),
    medianprops=dict(color='k', lw=1))


bp = MHW_chl_matched.loc[:,['MHW','DMS']].boxplot(by='MHW',
                                            grid=False,
                                            # positions=groups-space,
                                            positions=np.array([-0.15, 0.15]),
                                            ax=ax3,
                                            return_type='both',
                                            **box_param)

colors = ['lightgray', 'red']
for row_key, (ax,row) in bp.items():
    for i,box in enumerate(row['boxes']):
        box.set_facecolor(colors[i])



bp = MHW_chl_matched.loc[:,['MHW','DMS:chl']].boxplot(by='MHW',
                                            grid=False,
                                            # positions=groups+space,
                                            positions=np.array([0.85, 1.15]),
                                            # color='orange',
                                            ax=ax_1,
                                            return_type='both',
                                            **box_param)

colors = ['lightgray', 'red']
for row_key, (ax,row) in bp.items():
    for i,box in enumerate(row['boxes']):
        box.set_facecolor(colors[i])
        
ax3.legend(bp[0][1]["boxes"], 
           ['Ambient SST', 'Category 1 \nMHW (Moderate)'], 
           loc='center left',
           fontsize=14,
           bbox_to_anchor=(0,0.15,0.5,1))


bbox = dict(facecolor='w', ec='None')
sig_fontsize=20
lw = 1
dh = 0.1
barh = 0.05
barplot_annotate_brackets(
    ax3,
    0,
    1,
    '***',
    center=[-0.15, 0.15, 0.85, 1.15],
    height=np.tile(72.7, 4),
    # [10,10,10,10],
    bbox=bbox,
    barh=barh,
    dh=dh,
    lw=lw,
    fs=sig_fontsize)


dh = 0.1
barh = 0.2
barplot_annotate_brackets(
    ax_1,
    2,
    3,
    '***',
    center=[-0.15, 0.15, 0.85, 1.15],
    height=np.tile(300, 4),
    bbox=bbox,
    barh=barh,
    dh=dh,
    lw=lw,
    fs=sig_fontsize)


ax3.set_xticks(groups)
ax3.set_xticklabels(['DMS', 'DMS:chl-a'])
ax3.set_xlabel('')

ax3.set_ylabel('DMS (nM)', color='k')
ax_1.set_ylabel('DMS:chl-a (nmol $\mu$g$^{-1}$)', color='k')

yticks_fmt = dict(axis='y', labelsize=24)
ax3.tick_params(colors='k', **yticks_fmt)
ax_1.tick_params(colors='k', **yticks_fmt)

ax3.set_ylim(0,90)
ax_1.set_ylim(0,360)

ax3.set_title('')
ax_1.set_title('')
fig.texts = [] #flush the old super titles

ax3.text(0.03,0.9, 'b', color='k', fontweight='bold', fontsize=24,
          transform=ax3.transAxes)

#------------------------------------------------------------------------------
fig.subplots_adjust(wspace=0.4)

fig.savefig(fig_dir+'Fig_2.tif', bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(fig_dir+'Fig_2.png', bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})

#%% FIG 3: plot along transect SSN anomalies, Fv/Fm, and DMS/O/P turnover rates

min_lon, max_lon, min_lat, max_lat = -146, -123, 48, 52
ms = 30

#------------------------------------------------------------------------------
fig = plt.figure(figsize=(6,9), dpi=300)
font={'family':'sans-serif',
      'weight':'normal',
      'size':'18'} 
plt.rc('font', **font) # sets the specified font formatting globally
gs1 = fig.add_gridspec(20, 5, hspace=0.1)
gs2 = fig.add_gridspec(20, 5, hspace=0.8)
gs3 = fig.add_gridspec(20, 5, hspace=0.1)

# main plots
ax_b = fig.add_subplot(gs1[0,0:5])

ax2 = fig.add_subplot(gs2[1:5,0:5]) 
ax3 = fig.add_subplot(gs2[5:10,0:5]) 
ax4 = fig.add_subplot(gs2[10:15,0:5]) 
ax5_b = fig.add_subplot(gs2[15,0:5]) 

ax5 = fig.add_subplot(gs3[16:21,0:5]) 

FvFm_color = 'r'
NPQ_color = 'k'
NPQ_metric = 'NPQ$_{sv}$'
width = 0.01
FvFm_lims = (0, 0.6)
NPQ_lims = (0, 1)
newcmap = cmocean.tools.crop_by_percent(cmocean.cm.balance_r, 10, which='both', N=None)
colors = newcmap(np.array([0.1,0.6,0.8]))
FRRF_colors = newcmap(np.array([0.1,0.8]))

width = 1.2 

fontsize=10
lw=0.5
stations = [
    'P26', 
    'P35',
    'P25', 
    'P24',
    'P23',
    'P22',
    'P21',
    'P20',
    'P19',
    'P18',
    'P17', 
    'P16',
    'P15',
    'P14',
    'P13',
    'P12',
    'P11',
    'P10',
    'P9', 
    'P8',
    'P7',
    'P6',
    'P5',
    'P4', 
    'P3', 
    'P2',
    'P1',]
#==============================================================================
#### Plot SSN data

alpha = 0.7
boxprops = dict(linestyle='-', linewidth=lw, color='grey', alpha=alpha)
medianprops = dict(linestyle='-', linewidth=lw, color='grey', alpha=alpha)
whiskerprops = dict(linestyle='-', linewidth=lw, color='grey', alpha=alpha)
capprops = dict(linestyle='-', linewidth=lw, color='grey', alpha=alpha)
flierprops = dict(linestyle='None', ms=3, linewidth=lw, color='grey', alpha=alpha)

SSN_mean.loc[:,stations].boxplot(
    ax=ax2,
    positions=SSN.loc[stations].index.get_level_values('lon'),
    widths=0.4,
    grid=False,
    boxprops=boxprops,
    medianprops=medianprops,
    whiskerprops=whiskerprops,
    capprops=capprops,
    flierprops=flierprops
    )
ax_x = ax_b.twiny()

SSN_mean.loc[:,stations].boxplot(
    ax=ax_x,
    positions=SSN.loc[stations].index.get_level_values('lon'),
    widths=0.4,
    grid=False,
    boxprops=boxprops,
    medianprops=medianprops,
    whiskerprops=whiskerprops,
    capprops=capprops,
    flierprops=flierprops,
    )

ax2.plot(SSN.loc[stations].index.get_level_values('lon'),
        SSN_mean.loc[:,stations].mean(axis=0),
        ls='--',
        lw=lw,
        c='k',
        label='Historical Mean (2007-2021)')

ax2.plot(
    SSN.loc[stations].index.get_level_values('lon'),
    SSN.loc[stations].values,
    marker='.',
    c='r',
    markeredgecolor='k',
    ls='--',
    ms=10,
    lw=lw,
    label='Aug 2022')



ax2.fill_between(SSN.loc[stations].index.get_level_values('lon'),
                SSN_mean.loc[:,stations].mean(axis=0),
                SSN.loc[stations, 'SSN'].values,
                color='lightgrey',
                alpha=0.3)


ax_b.set_ylim(16,25)
ax_b.set_xticklabels([])
ax2.set_ylim(-0.2,15)
ax2.set_xlabel('')

ax2.set_ylabel('Sea Surface Nitrate \n(SSN, $\mu$M)', fontsize=fontsize)

d = .25  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax_b.plot([0, 1], [0, 0], transform=ax_b.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

ax2.spines[['top']].set_visible(False)
ax_b.spines[['bottom']].set_visible(False)
ax_x.spines[['bottom']].set_visible(False)
ax_b.tick_params(axis='x', which='both', length=0)

ax2.tick_params(which='both', labelsize=fontsize)
ax_b.tick_params(which='both', labelsize=fontsize)
ax_x.tick_params(which='both', labelsize=fontsize-2, rotation=90)

ax2.set_xticks(np.arange(-125,-145-2.5, -2.5)[::-1])

ax2.legend(
    loc='upper center',
    fontsize=fontsize-2, 
    bbox_to_anchor=(0.5, 1.25),)

ax_b.text(0.02,0.2, 'a', color='k', fontweight='bold', fontsize=18,
         transform=ax_b.transAxes)

#==============================================================================
#### Plot photo-physiology (Fv/Fm, NPQ)
ax3.bar(FvFm_mean.index.get_level_values('lon'),
        FvFm_mean.values[:,0],
        width=width,
        lw=lw,
        color=FRRF_colors[0,:],
        ec='k')

ax3.errorbar(FvFm_mean.index.get_level_values('lon'),
             FvFm_mean.values[:,0],
             yerr=FvFm_sd.values[:,0],
             linestyle='None',
             capsize=2,
             lw=lw,
             c='k',)

ax3.text(LineP_stn_coords[1,1],
         ax3.get_ylim()[1]*0.1,
          'n.d.',
          ma='center',
          ha='center',
          va='center',
          fontsize=fontsize)

ax3.text(LineP_stn_coords[2,1],
         ax3.get_ylim()[1]*0.1,
          'n.d.',
          ma='center',
          ha='center',
          va='center',
          fontsize=fontsize)

ax3.set_xticklabels([])
ax3.set_ylabel(r'F$_{\rm v}$/F$_{\rm m}$', fontsize=fontsize)
ax3.set_ylim(0,0.6)

ax3.tick_params(which='both', labelsize=fontsize)

ax3.text(0.02,0.83, 'b', color='k', fontweight='bold', fontsize=18,
          transform=ax3.transAxes)
#==============================================================================
#### Plot native DMS turnover rates

#------------------------------------------------------------------------------
ax4.bar(DMS_nat_rates['lon'],
        DMS_nat_rates['63_conc_nM'],
        width=width,
        color='grey',
        lw=lw,
        ec='k',)

ax4.errorbar(DMS_nat_rates['lon'],
             DMS_nat_rates['63_conc_nM'],
             yerr=DMS_nat_rates_sd['63_conc_nM'],
             linestyle='None',
             capsize=2,
             lw=lw,
             c='k',)

ax4.text(LineP_stn_coords[2,1],
         ax4.get_ylim()[1]*0.05,
          'n.d.',
          ma='center',
          ha='center',
          va='center',
          fontsize=fontsize)

ax4.set_ylabel('DMS Turnover (hr$^{\mathrm{-1}}$)', fontsize=fontsize)
ax4.set_xticklabels([])

ax4.tick_params(which='both', labelsize=fontsize)

ax4.text(0.02,0.85, 'c', color='k', fontweight='bold', fontsize=18,
          transform=ax4.transAxes)

#-------------------------------------------------------------------------------
#### Plot tracer turnover rates
lgd_labels = ['D3-DMS Consumption', 'D6-DMSP Cleavage', 'D6,$^{13}$C$_{2}$-DMSO Reduction']
width = width/3
widths = [-width, 0, width]
for i in range(len(DMS_tracer_rates)):
    for j, tracer in enumerate(DMS_tracer_rates.iloc[:,2:].columns):
        if i==0:
            label = lgd_labels[j]
        else:
            label = None
        ax5.bar(DMS_tracer_rates.loc[:,'lon'].iloc[i] + widths[j],
                DMS_tracer_rates.loc[:,tracer].iloc[i],
                width=width,
                color=colors[j],
                lw=lw,
                alpha=1,
                label=label,
                ec='k',)
        ax5.errorbar(DMS_tracer_rates.loc[:,'lon'].iloc[i] + widths[j],
                     DMS_tracer_rates.loc[:,tracer].iloc[i],
                     yerr=DMS_tracer_rates_sd.loc[:,tracer].iloc[i],
                     linestyle='None',
                     capsize=2,
                     lw=lw,
                     label=None,
                     c='k',)
        # add broken axis data
        ax5_b.bar(DMS_tracer_rates.loc[:,'lon'].iloc[i] + widths[j],
                DMS_tracer_rates.loc[:,tracer].iloc[i],
                width=width,
                color=colors[j],
                lw=lw,
                alpha=1,
                label=label,
                ec='k',)
        ax5_b.errorbar(DMS_tracer_rates.loc[:,'lon'].iloc[i] + widths[j],
                     DMS_tracer_rates.loc[:,tracer].iloc[i],
                     yerr=DMS_tracer_rates_sd.loc[:,tracer].iloc[i],
                     linestyle='None',
                     capsize=2,
                     lw=lw,
                     label=None,
                     c='k',)

ax5.set_ylim(-0.035, 0.15)
ax5_b.set_ylim(0.24,0.4)

ax5.text(LineP_stn_coords[2,1],
         ax5_b.get_ylim()[1]*0.03,
          'n.d.',
          ma='center',
          ha='center',
          va='center',
          fontsize=fontsize)

d = .25  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax5_b.plot([0, 1], [0, 0], transform=ax5_b.transAxes, **kwargs)
ax5.plot([0, 1], [1, 1], transform=ax5.transAxes, **kwargs)

ax5.spines[['top']].set_visible(False)
ax5_b.spines[['bottom']].set_visible(False)
ax5_b.tick_params(axis='x', which='both', length=0)
ax5_b.set_xticks([])



lgd = ax5.legend(
    title='Turnover Pathway',
    fontsize=fontsize-2,
    loc='upper center',
    bbox_to_anchor=(0.78, 1.2),
    )

plt.setp(lgd.get_title(),fontsize=fontsize-2)

ax5.axhline(0, color='k', ls='--', lw=lw)
ax5.set_xlabel('')
ax5.set_ylabel(' Turnover rate \nconstant (hr$^{\mathrm{-1}}$)', fontsize=fontsize)
ax5_b.text(0.02,-0.2, 'd', color='k', fontweight='bold', fontsize=18,
          transform=ax5_b.transAxes)

ax5.tick_params(which='both', labelsize=fontsize)
ax5_b.tick_params(which='both', labelsize=fontsize)

ax5.set_xlabel('Longitude ($^{o}$W)', fontsize=fontsize)
#------------------------------------------------------------------------------

ax2.set_xlim(-145.6, -124.99917)
ax3.set_xlim(-145.6, -124.99917)
ax4.set_xlim(-145.6, -124.99917)
ax5.set_xlim(-145.6, -124.99917)
ax5_b.set_xlim(-145.6, -124.99917)

fig.savefig(fig_dir+'Fig_3.tif', bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})

#%% FIG 4: Plot along transect wind speed & MLD

# P26_wind_ind= 207

# wind_min_bound = DMS_wind_ship_anom.iloc[:P26_wind_ind,:]['wind'][DMS_wind_ship_anom.iloc[:P26_wind_ind,:]['wind']>6].index.get_level_values('lonbins').min()
# wind_max_bound = DMS_wind_ship_anom.iloc[:P26_wind_ind,:]['wind'][DMS_wind_ship_anom.iloc[:P26_wind_ind,:]['wind']>6].index.get_level_values('lonbins').max()

fig = plt.figure(figsize=(6,9), dpi=300)
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ms = 12
lw = 0.5
fontsize=12
panel_lettering = 18


#------------------------------------------------------------------------------
# apply a 6-point running average
window = 6
wind_smooth = np.convolve(DMS_wind_ship_anom.loc[:,'wind'], np.ones(window)/window, mode='valid')
wind_smooth_lons = np.convolve(DMS_wind_ship_anom.index.get_level_values('lons').values, np.ones(window)/window, mode='valid')
# get the index for P26
P26_wind_ind = wind_smooth_lons.argmin()

ax.plot(wind_smooth_lons[:P26_wind_ind-window],
        wind_smooth[:P26_wind_ind-window],
        color='k',
        ls='--',
        lw=lw,
        marker='.',
        ms=ms*0.75)

ax.set_ylabel('Wind Speed \n(m s$^{-1}$)', fontsize=fontsize)

ax.tick_params(axis='both', labelsize=fontsize)

ax.text(0.01,0.9, 'a', color='k', fontweight='bold', fontsize=panel_lettering,
         transform=ax.transAxes)
#------------------------------------------------------------------------------
stations = [
    'P1',
    'P2',
    'P3',
    'P4',
    'P5',
    'P6',
    'P7',
    'P8',
    'P9',
    'P10',
    'P11',
    'P12',
    'P14',
    'P15',
    'P16',
    'P17',
    'P18',
    'P19',
    'P20',
    'P21',
    'P22',
    'P23',
    'P24',
    'P25',
    'P35',
    'P26',
]

alpha = 0.7
boxprops = dict(linestyle='-', linewidth=lw, color='grey', alpha=alpha)
medianprops = dict(linestyle='-', linewidth=lw, color='grey', alpha=alpha)
whiskerprops = dict(linestyle='-', linewidth=lw, color='grey', alpha=alpha)
capprops = dict(linestyle='-', linewidth=lw, color='grey', alpha=alpha)
flierprops = dict(linestyle='None', ms=5, linewidth=lw, color='grey', alpha=alpha)


ax_2 = ax2.twiny()
MLD_mean.loc[:,stations].boxplot(ax=ax_2,
                                  grid=False,
                                  widths=0.4,
                                  positions=MLDs.loc[stations].index.get_level_values('lon'),
                                  boxprops=boxprops,
                                  medianprops=medianprops,
                                  whiskerprops=whiskerprops,
                                  capprops=capprops,
                                  flierprops=flierprops)



ax2.plot(MLDs.loc[stations].index.get_level_values('lon'),
        MLD_mean.loc[:,stations].mean(),
        ms=ms,
        ls='--',
        lw=lw,
        c='grey',
        label='Historical mean (2007-2021)')

ax2.plot(MLDs.loc[stations].index.get_level_values('lon'),
        MLDs.loc[stations],
        marker='.',
        c='r',
        markeredgecolor='k',
        ls='--',
        lw=lw,
        ms=ms,
        label='Aug 2022')

ax2.fill_between(MLDs.loc[stations].index.get_level_values('lon'),
                MLD_mean.loc[:,stations].mean(),
                MLDs.loc[stations].squeeze(),
                color='lightgrey',
                alpha=0.3)


ax_2.invert_yaxis()
ax2.set_ylabel('MLD (m)', fontsize=fontsize)
ax2.legend(fontsize=10, loc='upper center')
ax2.set_ylim(0,45)
ax2.invert_yaxis()
ax_2.tick_params(axis='x', rotation=90, labelsize=8)

ax2.tick_params(axis='both', labelsize=fontsize)

ax2.set_xlabel('Longitude ($^{o}$W)', fontsize=fontsize)

ax2.text(0.01,0.9, 'b', color='k', fontweight='bold', fontsize=panel_lettering,
         transform=ax2.transAxes)
#------------------------------------------------------------------------------
xlims = (-145.3, -124.5)
ax.set_xlim(xlims)
ax2.set_xlim(xlims)
ax_2.set_xlim(xlims)

xtick_lons = np.arange(-125,-145-5, -5)
ax.set_xticks(xtick_lons)
ax2.set_xticks(xtick_lons)

ax.set_xticklabels([])

fig.savefig(fig_dir+'Fig_4.tif', bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})

#%% ED FIG 1: subpolar gyre SST anomaly (Aug 10-24, 2022)

keys_ = np.arange(10,25)
keys_ = [str(i) for i in keys_]

SST_to_plot = SST_anom_full.loc[keys_].groupby('lat').mean()

fig = plt.figure(figsize=(12,12), dpi=300)
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
h = ax.pcolormesh(
    SST_to_plot.loc[30:62, -180:-123].columns.values,
    SST_to_plot.loc[30:62, -180:-123].index.values,
    SST_to_plot.loc[30:62, -180:-123].values,
    vmin=1,
    vmax=5,
    cmap=mpl.cm.Reds,
    zorder=1,
    transform=ccrs.PlateCarree())

h2 = ax.scatter(
    uw_DMS_ind.loc[:DMS_OSP_loc].loc[:,'lon'],
    uw_DMS_ind.loc[:DMS_OSP_loc].loc[:,'lat'],
    s=5,
    vmin=0,
    vmax=30,
    c='k',
    zorder=3,
    transform=ccrs.PlateCarree())

for stn in LineP_stns.index[1:]:
    ax.scatter(
        LineP_stns['lon'].loc[stn],
        LineP_stns['lat'].loc[stn],
        s=40,
        c='None',
        ec='k',
        marker='o',
        zorder=3,
        transform=ccrs.PlateCarree())
    ax.text(
        x=LineP_stns['lon'].loc[stn],
        y=LineP_stns['lat'].loc[stn]-0.4,
        s=stn,
        ha='center',
        va='top',
        fontsize=18,
        zorder=3,
        transform=ccrs.PlateCarree())

gl = ax.gridlines(draw_labels=True, zorder=2)
gl.top_labels = False
gl.right_labels = False
ax.add_feature(cartopy.feature.LAND, edgecolor='None', facecolor='darkgray', zorder=2)
ax.add_feature(cartopy.feature.COASTLINE, edgecolor='k', zorder=2)
ax.set_extent([-179, -123, 35, 58])

divider = make_axes_locatable(ax)
ax_cb = divider.new_horizontal(size="2%", pad=0.2, axes_class=plt.Axes)
fig.add_axes(ax_cb)
cb1 = plt.colorbar(h, cax=ax_cb, extend='max')
cb1.ax.tick_params(labelsize=20)
ax_cb.xaxis.set_ticks_position('top')
ax_cb.xaxis.set_label_position('top')
cb1.set_label('SST Anomaly ($^{o}$C)', fontsize=20, labelpad=10)

fig.savefig(fig_dir+'Fig_S1.tif', bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})

#%% ED FIG 2: Plot relative species abundance along Line P
#------------------------------------------------------------------------------

dpi=300
fig = plt.figure(figsize=(8.75,8.75), dpi=dpi)
ax = fig.add_subplot(111)
lw=0.5
fontsize=8

stations = [
    'Haro59',
    'JF2',
    'P1',
    'P2',
    'P3',
    'P4',
    'P5',
    'P6',
    'P7',
    'P8',
    'P9',
    'P10',
    'P11',
    'P12',
    'P14',
    'P15',
    'P16',
    'P17',
    'P18',
    'P19',
    'P20',
    'P21',
    'P22',
    'P23',
    'P24',
    'P25',
    'P35',
    'P26',
]

#------------------------------------------------------------------------------
widths = np.concatenate((np.tile(0.6,2),
                         np.tile(0.25,4),
                         np.tile(0.4,8),
                         np.tile(0.6,14)))

colors = plt.cm.jet(np.linspace(0,1,len(taxa_per.columns)))
for i, taxa in enumerate(taxa_per.columns):
    if i == 0:
        bottom = 0
    else:
        bottom = taxa_per.loc[stations].iloc[:,0:i].sum(axis=1)
    
    ax.bar(
        MLDs.loc[stations].index.get_level_values('lon'),
        height=taxa_per.loc[stations, taxa],
        bottom=bottom,
        width=widths,
        color=colors[i],
        lw=lw,
        alpha=1,
        label=taxa,
        ec='k',)
    
lineP_ax = ax.secondary_xaxis('top')
lineP_ax.set_xticks(MLDs.loc[stations].index.get_level_values('lon').values)
lineP_ax.set_xticklabels(MLDs.loc[stations].index.get_level_values('station').values, fontdict={'fontsize':10})
lineP_ax.tick_params(axis='x', rotation=90)

ax.set_ylim(0,100)

h,l = ax.get_legend_handles_labels()
lgd = ax.legend(h,
          l,
          loc='upper right',
          bbox_to_anchor=(1,0,0.25,1.01),
          title='Phytoplankton Class',
          fontsize=fontsize+2,
          title_fontsize=fontsize+2,
          edgecolor='k',
          )
lgd.get_frame().set_linewidth(lw)

ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
ax.tick_params(which='both', labelsize=12)
ax.set_xlabel('Longitude ($^{o}$W)', fontsize=12)

ax.set_xlim(-145.6, -124.99917)

fig.savefig(fig_dir+'Fig_S2.tif', bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})

#%% ED FIG 3: Plot Brunt-vaisala frequencies

#------------------------------------------------------------------------------
stations = ['P4','P12','P16','P20','P26']
fig = plt.figure(figsize=(12,6), dpi=300)
gs = fig.add_gridspec(2, 5)
axes = {}
for i in range(len(stations)):
    axes[i] = fig.add_subplot(gs[0,i])

#------------------------------------------------------------------------------
lon_n = 0.1
total_depth = 60
start_depth = 1
depth_n = 1

for i,stn in enumerate(stations[::-1]):
    axes[i].plot(N_sq.loc[idx[:,:,:,:,stn]].sort_index(level='Pressure:CTD [dbar]', axis=0),
           N_sq.loc[idx[:,:,:,:,stn]].sort_index(level='Pressure:CTD [dbar]', axis=0).index.get_level_values('Pressure:CTD [dbar]'),
           marker='.',
           ls='--',
           c='k',
           lw=1,
           ms=5,
           label=stn,
           )
    axes[i].invert_yaxis()
    axes[i].set_ylim(total_depth,0)
    axes[i].xaxis.tick_top()
    axes[i].xaxis.set_label_position('top')
    axes[i].spines[['right', 'bottom',]].set_visible(False)
    axes[i].set_title(stn, fontsize=16)
    axes[i].set_xlim(0,0.005)
    axes[i].ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
    axes[i].xaxis.offsetText.set_fontsize(14)
    if i != 0:
        axes[i].set_yticklabels([])
    else:
        axes[i].set_ylabel('Depth (m)', fontsize=12)

    axes[i].tick_params(which='both', labelsize=12)

stations = N_sq.index.get_level_values('LOC:STATION').unique()

#------------------------------------------------------------------------------
interp_lons = np.arange(np.floor(N_sq.loc[idx[:,:,:,start_depth:total_depth,stations]].index.get_level_values('LOC:LONGITUDE').values.min()),
                        np.ceil(N_sq.loc[idx[:,:,:,start_depth:total_depth,stations]].index.get_level_values('LOC:LONGITUDE').values.max())+lon_n,lon_n)

interp_depths = np.arange(start_depth,total_depth+depth_n,depth_n)
xx,yy = np.meshgrid(interp_lons,
                    interp_depths) # coordinate pts
coords = np.stack([xx.flatten(), yy.flatten()], axis=1)

interpd = scipy.interpolate.griddata(np.stack([N_sq.loc[idx[:,:,:,start_depth:total_depth,stations]].index.get_level_values('LOC:LONGITUDE').values,
                                               N_sq.loc[idx[:,:,:,start_depth:total_depth,stations]].index.get_level_values('Pressure:CTD [dbar]').values],axis=1),
                                      N_sq.loc[idx[:,:,:,start_depth:total_depth,stations]].values,
                                      (xx,yy),
                                      method='linear')

interpd = pd.DataFrame(interpd, index=interp_depths, columns=(interp_lons+(0.1*lon_n)))
interpd = interpd.loc[:,-145.1:-120.5]
#------------------------------------------------------------------------------

ax = fig.add_subplot(gs[1,0:6])

vmin = N_sq.loc[idx[:,:,:,:,stations]].min()
vmax = N_sq.loc[idx[:,:,:,:,stations]].max()
cmap = cmocean.cm.thermal

h = ax.contourf(interpd.columns.values,
            interpd.index.values,
            interpd.values,
            levels=100,
            # norm=norm,
            cmap=cmap)

ax.scatter(
    N_sq.loc[idx[:,:,:,:,stations]].index.get_level_values('LOC:LONGITUDE').values,
    N_sq.loc[idx[:,:,:,:,stations]].index.get_level_values('Pressure:CTD [dbar]').values,
    s=1,
    c=N_sq.loc[idx[:,:,:,:,stations]].values,
    ec='k',
    linewidths=0.5,
    cmap=cmap)


ax.plot(MLDs.loc[stations].sort_index(level='lon').index.get_level_values('lon').values,
        MLDs.loc[stations].sort_index(level='lon').values[:,0],
        'w.--',
        lw=1,
        ms=5,
)


ax.tick_params('both', length=7)
ax.invert_yaxis()
ax.set_ylim(total_depth,0)
ax.set_ylabel('Depth (m)', fontsize=12)
ax.set_xlabel('Longitude ($^{o}$W)', fontsize=12)
ax.tick_params(which='both', labelsize=12)

divider = make_axes_locatable(ax)
ax_cb = divider.new_horizontal(size="2%", pad=0.2, axes_class=plt.Axes)
fig.add_axes(ax_cb)
cb1 = plt.colorbar(h, cax=ax_cb)
cb1.ax.tick_params(labelsize=12)
ax_cb.xaxis.set_ticks_position('top')
ax_cb.xaxis.set_label_position('top')
cb1.set_label('N$^{2}$', fontsize=12, labelpad=10)
cb1.formatter.set_powerlimits((0, 0))
# to get 10^3 instead of 1e3
cb1.formatter.set_useMathText(True)
# and set its fontsize
cb1.ax.yaxis.get_offset_text().set_fontsize(12)

stations = ['P4','P12','P16','P20','P26']

lineP_ax = ax.secondary_xaxis('top')
lineP_ax.set_xticks(SSN.loc[stations].index.get_level_values('lon'))
lineP_ax.set_xticklabels(SSN.loc[stations].index.get_level_values('station'), fontdict={'fontsize':12})
lineP_ax.tick_params(which='both', pad=0.1)

for x in lineP_ax.get_xticks():
    ax.axvline(x, color='gray',
                zorder=-1,
                linestyle="-.",
                linewidth=0.5,
                dashes=(5, 10),
                )

fig.savefig(fig_dir+'Fig_S3.tif', bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})

#%% ED FIG 4: T/S section plots

lon_n = 0.1
total_depth = 60
start_depth = 2
depth_n = 1

interp_lons = np.arange(np.floor(temp_depth_anom.index.get_level_values('lon').values.min()),
                        np.ceil(temp_depth_anom.index.get_level_values('lon').values.max())+lon_n,lon_n)

interp_depths = np.arange(start_depth,total_depth+depth_n,depth_n)
xx,yy = np.meshgrid(interp_lons,
                    interp_depths) # coordinate pts
coords = np.stack([xx.flatten(), yy.flatten()], axis=1)

interp_method = 'cubic'
#------------------------------------------------------------------------------
#### interpolate temp anomalies by longitude x depth
interpd = scipy.interpolate.griddata(np.stack([temp_depth_anom.loc[idx[start_depth:total_depth]].index.get_level_values('lon').values,
                                               temp_depth_anom.loc[idx[start_depth:total_depth]].index.get_level_values('depth').values],axis=1),
                                      temp_depth_anom.loc[idx[start_depth:total_depth]].values,
                                      (xx,yy),
                                      method=interp_method)

temp_interpd = pd.DataFrame(interpd, index=interp_depths, columns=(interp_lons+(0.1*lon_n)))


# interpolate salinity by longitude x depth
interpd = scipy.interpolate.griddata(np.stack([sal_depth_anom.loc[idx[start_depth:total_depth]].index.get_level_values('lon').values,
                                               sal_depth_anom.loc[idx[start_depth:total_depth]].index.get_level_values('depth').values],axis=1),
                                      sal_depth_anom.loc[idx[start_depth:total_depth]].values,
                                      (xx,yy),
                                      method=interp_method)

sal_interpd = pd.DataFrame(interpd, index=interp_depths, columns=(interp_lons+(0.1*lon_n)))

#==============================================================================
# extracted matching stations between Aug 2022 CTD profiles and historic profiles
stations = ['P26', 'PA-016', 'P35', 'P25', 'P24', 'P23', 'P22', 'P21', 'P20', 'P19',
       'P18', 'P17', 'P16', 'P15', 'P14', 'P13', 'P12', 'P11', 'P10',
       'P9', 'P8', 'P7', 'P6', 'P5', 'P4', 'P3', 'P2',
       'LaP Mooring', 'P1',]

fig = plt.figure(figsize=(12,12), dpi=300)
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

#------------------------------------------------------------------------------
vmin = -4
vmax = 4
norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) # scales to accentuate depth colors, and diverge at 0
h = ax.pcolormesh(temp_interpd.columns,
              temp_interpd.index,
              temp_interpd,
              norm=norm,
              cmap=plt.cm.RdBu_r)
ax.contourf(temp_interpd.columns,
              temp_interpd.index,
              temp_interpd,
              levels=100,
              norm=norm,
              cmap=plt.cm.RdBu_r)

ax.scatter(temp_depth_anom.index.get_level_values('lon'),
            temp_depth_anom.index.get_level_values('depth'),
            s=1,
            c='k')

ax.plot(MLDs.index.get_level_values('lon'),
        MLDs.values,
        ls='--',
        c='k',
        lw=2)

ax.set_ylim(0,total_depth)
ax.invert_yaxis()

divider = make_axes_locatable(ax)
ax_cb = divider.new_horizontal(size="2%", pad=0.2, axes_class=plt.Axes)
fig.add_axes(ax_cb)
cb1 = plt.colorbar(h, cax=ax_cb)
cb1.set_label('Temperature Anomaly ($^{o}$C)', fontsize=20, labelpad=10)

ax.set_xlim(-145.1,-125.4)

lineP_ax = ax.secondary_xaxis('top')
lineP_ax.set_xticks(SSTA_depth_lons.values[:,0])
lineP_ax.set_xticklabels(SSTA_depth_lons.index.get_level_values('station'), fontdict={'fontsize':10})
lineP_ax.tick_params(axis='x', rotation=90)

# Create offset transform by 5 points in x direction # from here: https://stackoverflow.com/questions/28615887/how-to-move-a-tick-label
dx = 3/72.; dy = 0/72. 
offset = mpl.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
for i, label in enumerate(lineP_ax.xaxis.get_majorticklabels()):
    if i in {1,27}:
        label.set_transform(label.get_transform() + offset)

ax.set_ylabel('Depth (m)')
ax.set_xticklabels([])

ax.text(-0.05,1.05, 'a', color='k', fontweight='bold', fontsize=24,
         transform=ax.transAxes)
#------------------------------------------------------------------------------
vmin = -0.5
vmax = 0.5
norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) # scales to accentuate depth colors, and diverge at 0
h2 = ax2.pcolormesh(sal_interpd.columns,
              sal_interpd.index,
              sal_interpd,
              norm=norm,
              cmap=plt.cm.RdBu_r)
ax2.contourf(sal_interpd.columns,
              sal_interpd.index,
              sal_interpd,
               levels=1000,
              norm=norm,
              cmap=plt.cm.RdBu_r)

ax2.scatter(sal_depth_anom.index.get_level_values('lon'),
            sal_depth_anom.index.get_level_values('depth'),
              s=1,
              c='k')

ax2.plot(MLDs.index.get_level_values('lon'),
        MLDs.values,
        'k--',
        lw=2)

ax2.set_ylim(0,total_depth)
ax2.invert_yaxis()

divider = make_axes_locatable(ax2)
ax_cb = divider.new_horizontal(size="2%", pad=0.2, axes_class=plt.Axes)
fig.add_axes(ax_cb)
cb1 = plt.colorbar(h2, cax=ax_cb)
cb1.set_label('Salinity Anomaly (PSS-78)', fontsize=20, labelpad=10)

ax2.set_xlim(-145.1,-125.4)
ax2.set_ylabel('Depth (m)')
ax2.set_xlabel('Longitude ($^{o}$W)')

ax2.text(-0.05,1.05, 'b', color='k', fontweight='bold', fontsize=24,
         transform=ax2.transAxes)

fig.savefig(fig_dir+'Fig_S4.tif', bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})

#%% ED FIG 5: Transect plots of DMS concentrations and sea-air flux

P26_wind_ind = 209
# P26_wind_ind= 199

fig = plt.figure(figsize=(12,12), dpi=300)
ax2 = fig.add_subplot(211)
ax3 = fig.add_subplot(212)

ax2.plot(DMS_wind_ship_anom.iloc[:P26_wind_ind,:].index.get_level_values('lons'),
        DMS_wind_ship_anom.iloc[:P26_wind_ind,:]['DMS'],
        color='k',
        ls='--',
        marker='.',
        ms=15)
ax2.set_ylabel('DMS (nM)')
ax2.set_xticklabels([])

ax2.text(0.01,0.9, 'a', color='k', fontweight='bold', fontsize=24,
         transform=ax2.transAxes)

#------------------------------------------------------------------------------

#### calculate sea-air flux (F_DMS)
k = 2.1*DMS_wind_ship_anom['wind']-2.8
DMS_wind_ship_anom['F_DMS'] = k*DMS_wind_ship_anom['DMS']*0.24

ax3.plot(DMS_wind_ship_anom.iloc[:P26_wind_ind,:].index.get_level_values('lons'),
        DMS_wind_ship_anom.iloc[:P26_wind_ind,:]['F_DMS'],
        color='k',
        ls='--',
        marker='.',
        ms=15)
ax3.set_ylabel('Sea-air DMS flux \n(μmol m$^{−2}$ d$^{−1}$)')
ax3.set_xlabel('Longitude ($^{o}$W)')

ax3.text(0.01,0.9, 'b', color='k', fontweight='bold', fontsize=24,
         transform=ax3.transAxes)

fig.subplots_adjust(hspace=0.1)

fig.savefig(fig_dir+'Fig_S5.tif', bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})

#%% ED FIG 6: Xanthophyll cycling along Line P
#------------------------------------------------------------------------------

dpi=300
fig = plt.figure(figsize=(8.75,8.75/2), dpi=dpi)
ax = fig.add_subplot(211)
ax3 = fig.add_subplot(212)
lw=0.5
fontsize=10

stations = [
    'Haro59',
    'JF2',
    'P1',
    'P2',
    'P3',
    'P4',
    'P5',
    'P6',
    'P7',
    'P8',
    'P9',
    'P10',
    'P11',
    'P12',
    'P14',
    'P15',
    'P16',
    'P17',
    'P18',
    'P19',
    'P20',
    'P21',
    'P22',
    'P23',
    'P24',
    'P25',
    'P35',
    'P26',
]
#------------------------------------------------------------------------------
colors = ['lightgrey', 'black']

widths = np.concatenate((np.tile(0.6,2),
                         np.tile(0.25,4),
                         np.tile(0.4,8),
                         np.tile(0.6,14)))

for i, xphyll in enumerate(dd_dt_norm.columns):
    if i == 0:
        bottom = 0
    else:
        bottom = dd_dt_norm.loc[stations].iloc[:,0:i].sum(axis=1)
    
    ax.bar(
        MLDs.loc[stations].index.get_level_values('lon'),
        height=dd_dt_norm.loc[stations, xphyll],
        bottom=bottom,
        width=widths,
        color=colors[i],
        lw=lw,
        alpha=1,
        label=xphyll,
        ec='k',)

ax.set_xticklabels([])
ax.set_xlabel('')
ax.set_ylabel(r'(Dd,Dt):chl-a ($\mu$g L$^{-1}$)', fontsize=fontsize)

ax.tick_params(axis='both', labelsize=12)
ax.legend(loc='upper right', fontsize=12)

ax.text(0.01,0.8, 'a', color='k', fontweight='bold', fontsize=18,
         transform=ax.transAxes)

#------------------------------------------------------------------------------
ax3.plot(MLDs.loc[stations].index.get_level_values('lon'),
         de_epox1_mean.loc[stations],
         ls='--',
         marker='.',
         ms=10,
         c='k')

ax3.set_ylabel(r'$\frac{Dt}{Dt+Dd}$ ($\mu$g $\mu$g$^{-1}$)', fontsize=fontsize)
ax3.tick_params(axis='both', labelsize=12)
ax3.set_ylim(0,0.6)
ax3.set_xlabel('Longitude ($^{o}$W)', fontsize=12)

ax3.text(0.01,0.8, 'b', color='k', fontweight='bold', fontsize=18,
         transform=ax3.transAxes)
#------------------------------------------------------------------------------

lineP_ax = ax.secondary_xaxis('top')
lineP_ax.set_xticks(MLDs.loc[stations].index.get_level_values('lon').values)
lineP_ax.set_xticklabels(MLDs.loc[stations].index.get_level_values('station').values, fontdict={'fontsize':8})
lineP_ax.tick_params(axis='x', rotation=90)

ax.set_xlim(-146.2, -122.5)
ax3.set_xlim(-146.2, -122.5)

fig.savefig(fig_dir+'Fig_S6.tif', bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})