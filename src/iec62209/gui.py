import logging
import traceback

import matplotlib.pyplot as plt
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .work import Work


def csv_upload_dialog():
    sg.set_options(auto_size_buttons=True)
    filename = sg.popup_get_file(
        'Dataset to read',
        title='Dataset to read',
        no_window=True,
        file_types=(("CSV Files", "*.csv"), ("All Files", "*.*")))
    return filename

def json_upload_dialog():
    sg.set_options(auto_size_buttons=True)
    filename = sg.popup_get_file(
        'File to read',
        title='File to read',
        no_window=True,
        file_types=(("JSON Files", "*.json"), ("All Files", "*.*")))
    return filename

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')

def mainwin():
    # Tab 1 --------------------------------------------------------------------
    docstr1 = """
Generates a random latin 8-dimensional hypercube sample and saves the resulting csv to file. The 8 x-variables are:

  frequency (MHz), power (dB), peak to average ratio, bandwidth, distance (mm), angle (deg), x (mm), y (mm)

Two additional values are provided: antenna type, modulation name. The user is then responsible for providing the
resulting z-values, the sar deviations, as additional columns with any chosen label. Multiple columns can be added,
but only one per model can be used.
"""
    work1 = Work()
    work1.generate_sample(size=400, show=False, save_to=None)
    layout1 = [[sg.Text(docstr1)],
              [sg.Text('Sample Size:'), sg.Input(key='T1-Size', default_text='400', size=(4, 1)), sg.Button('Generate Sample', key='T1-Generate')],
              [sg.Text('Filename:'), sg.Input(key='T1-Filename', default_text='sample.csv', size=(20, 1)), sg.Button('Save Sample', key='T1-Save'), sg.Text('', key='T1-Outfile')],
              [sg.Table(key='T1-Table',
                  expand_y=True,
                  values=work1.data['sample'].data.values.tolist(),
                  headings=work1.data['sample'].data.columns.values.tolist(),
                  display_row_numbers=True,
                  num_rows=min(50, len(work1.data['sample'].data)),
                  auto_size_columns=True)]]

    # Tab 2 --------------------------------------------------------------------
    docstr2 = """
Builds a model, outputs the empirical (blue) and theoretical (red) semivariogram after rescaling to an isotropic space.

Nothing is to be done by the user. The system analyses geostatistical properties along each direction in the data space,
computes an invertible mapping that converts the space into an isotropic one. A global multi-directional semi-variogram
is then built on the transformed space. The blue values represent empirical variances computed as a function of distance
between points. In red, a gaussian variogram curve is then fitted to the empirical values: it defines the variance kernel
used for all subsequent interpolations. The histogram below provides the distribution of all distances between points.
"""
    work2 = Work()
    layout2 = [[sg.Text(docstr2)],
              [sg.Text('z-variable:'), sg.Input(key='T2-Zvar', default_text='sard', size=(10, 1)), sg.Button('Load Data', key='T2-Load'), sg.Text('', key='T2-Infile')],
              [sg.Button('Build Model', key='T2-Build')],
              [sg.Text('Filename:'), sg.Input(key='T2-Filename', default_text='model.json', size=(20, 1)), sg.Button('Save Model', key='T2-Save'), sg.Text('', key='T2-Outfile')],
              [sg.Canvas(key='T2-Canvas')]]

    # Tab 3 --------------------------------------------------------------------
    docstr3 = """
Performs the good fit test: passes if the NRMSE is below 25%.

This test measures the quality of the variogram fit: how well the red curve fits the blue values. The statistic used is
the normalized root mean square error (NRMSE) of the variances along distances: it is equal to the RMSE of the residuals
divided by the mean of the variances. Unlike the RMSE, the NRMSE does not depend on the scale of the model and provides
a more robust evaluation of the goodness of fit. A NRMSE above 0.25 means the variogram model does not fit the empirical
variances well enough. The last histogram shows the distribution of the absolute values of residuals.
    """
    work3 = Work()
    layout3 = [[sg.Text(docstr3)],
              [sg.Button('Load Model', key='T3-Load'), sg.Text('', key='T3-Infile')],
              [sg.Button('Perform Test', key='T3-Test'), sg.Text('', key='T3-Result')],
              [sg.Canvas(key='T3-Canvas')]]

    # Tab 4 --------------------------------------------------------------------
    docstr4 = """
The model is now being confirmed by performing statistical tests that ascertain that the residuals between the model
and the measured data are distributed according to the expected probability distribution. Those residuals are normalized
into a distribution that needs to be as close as possible to the standard normal distribution. The test results are
presented in terms of:

i) The Shapiro-Wilk hypothesis p-value, which must be at least equal to 0.05 for the normality test to pass.

ii) The qq location and scale which need to be in the range of [-1, 1] and [0.5, 1.5] respectively the test to pass.

The test is successful if and only if both i) and ii) pass.
"""

    work4 = Work()
    layout4 = [[sg.Text(docstr4)],
              [sg.Button('Load Model', key='T4-Load1'), sg.Text('', key='T4-Infile1')],
              [sg.Text('z-variable:'), sg.Input(key='T4-Zvar', default_text='sard', size=(10, 1)), sg.Button('Load Test Data', key='T4-Load2'), sg.Text('', key='T4-Infile2')],
              [sg.Button('Perform Test', key='T4-Test'), sg.Text('', key='T4-Result')],
              [sg.Canvas(key='T4-Canvas')]]

    # Tab 5 --------------------------------------------------------------------
    docstr5 = """
Performs space exploration using at most maxsize trajectories and outputs to file the most critical regions.

A valid model is used to explore the entire data space for potential regions that exceed the most permissible error.
This is done by a hybrid search trajectory and population-based algorithm where a population of search trajectories
evolves through a predetermined number of iterations (generations) in such a way that:

i) the elements of the population are pulled towards the most extreme regions of the data space,

ii) the elements of the population exert a repulsive force on each other. This ensure not all trajectories will be
lead to the same locations, but insted will evenly cover a region deemed critical,

iii) the resulting values have meaningful SAR coordinates.

The resulting coordinates, with the computed z-values and associated probabilities to pass the mpe value are outputed
as a csv file whose name is to be provided by the user. The population usually stabilizes after 8 iterations.
"""

    work5 = Work()
    work5.init_critsample()
    colwidths = list(map(lambda x: max(len(x),7), work5.data['critsample'].data))
    layout5 = [[sg.Text(docstr5)],
              [sg.Button('Load Model', key='T5-Load'), sg.Text('', key='T5-Infile')],
              [sg.Text('Iterations:'), sg.Input(key='T5-Niter', default_text='8', size=(3, 1)), sg.Button('Perform Search', key='T5-Search')],
              [sg.Text('Filename:'), sg.Input(key='T5-Filename', default_text='crit_sample.csv', size=(20, 1)), sg.Button('Save Sample', key='T5-Save'), sg.Text('', key='T5-Outfile')],
              [sg.Table(key='T5-Table',
                  expand_x=True,
                  expand_y=True,
                  values=work5.data['critsample'].data.values.tolist(),
                  headings=work5.data['critsample'].data.columns.values.tolist(),
                  display_row_numbers=False,
                  num_rows=min(50, len(work5.data['critsample'].data)),
                  auto_size_columns=False,
                  col_widths = colwidths)]]

    # Window -------------------------------------------------------------------

    tabgrp = [
                [sg.Text('IEC62209 Validation Procedure'), sg.Push(), sg.Button('Close')],
                [sg.TabGroup(
                    [[sg.Tab('Sampling', layout1), sg.Tab('Modeling', layout2,),
                        sg.Tab('Fitting', layout3,), sg.Tab('Confirmation', layout4,),
                        sg.Tab('Exploration', layout5,)]],
                    tab_location='topleft',
                    title_color='White', tab_background_color='Gray',
                    selected_title_color='White', selected_background_color='Blue',
                    border_width=6), ]]

    window = sg.Window('IEC62209 Validation', tabgrp, resizable=True, finalize=True)
    # add the plot
    t2_figure = plt.figure()
    t2_canvas = draw_figure(window['T2-Canvas'].TKCanvas, t2_figure)
    t3_figure = plt.figure()
    t3_canvas = draw_figure(window['T3-Canvas'].TKCanvas, t3_figure)
    t4_figure = plt.figure()
    t4_canvas = draw_figure(window['T4-Canvas'].TKCanvas, t4_figure)
    while True:
        event, values = window.read()
        # Tab 1
        if event == 'T1-Generate':
            try:
                sz = int(values['T1-Size'])
                sz = min(sz, 1000)
                work1.generate_sample(size=sz, show=False, save_to=None)
                window['T1-Table'].update(values=work1.data['sample'].data.values.tolist())
            except:
                window['T1-Table'].update(values=[])
                work1.clear()
                pass
        elif event == 'T1-Save':
            try:
                fn = values['T1-Filename']
                work1.save_sample(fn)
                window['T1-Outfile'].update('saved to: ' + fn)
            except:
                window['T1-Outfile'].update('error saving file')
                pass
        # Tab 2
        elif event == 'T2-Load':
            try:
                fn = csv_upload_dialog()
                zvar = values['T2-Zvar']
                work2.load_init_sample(fn, zvar)
                window['T2-Infile'].update('loaded from: ' + fn)
            except:
                window['T2-Infile'].update('no file loaded')
                work2.clear()
                pass
        elif event == 'T2-Build':
            try:
                work2.make_model(show=False)
                if t2_canvas is not None:
                    t2_canvas.get_tk_widget().forget()
                t2_figure = work2.plot_model()
                if t2_figure is None:
                    raise RuntimeError()
                t2_canvas = draw_figure(window['T2-Canvas'].TKCanvas, t2_figure)
            except:
                if t2_figure is not None:
                    t2_figure.clear()
                    t2_canvas = draw_figure(window['T2-Canvas'].TKCanvas, t2_figure)
                work2.clear_model()
                pass
        elif event == 'T2-Save':
            try:
                fn = values['T2-Filename']
                work2.save_model(fn)
                window['T2-Outfile'].update('saved to: ' + fn)
            except:
                window['T2-Outfile'].update('error saving file')
                pass
        # Tab 3
        elif event == 'T3-Load':
            try:
                fn = json_upload_dialog()
                work3.load_model(filename=fn)
                window['T3-Infile'].update('loaded from: ' + fn)
            except:
                window['T3-Infile'].update('no file loaded')
                work3.clear()
                pass
        elif event == 'T3-Test':
            try:
                if t3_canvas is not None:
                    t3_canvas.get_tk_widget().forget()
                t3_figure = work3.goodfit_plot()
                if t3_figure is None:
                    raise RuntimeError()
                t3_canvas = draw_figure(window['T3-Canvas'].TKCanvas, t3_figure)
                gfres = work3.goodfit_test()
                window['T3-Result'].update(f'pass = {str(gfres[0]).lower()}, nrmse = {gfres[1]:.3f}')
            except:
                if t3_figure is not None:
                    t3_figure.clear()
                    t3_canvas = draw_figure(window['T3-Canvas'].TKCanvas, t3_figure)
                work3.clear_model()
                window['T3-Result'].update('')
                pass
        # Tab 4
        elif event == 'T4-Load1':
            try:
                fn = json_upload_dialog()
                work4.load_model(filename=fn)
                window['T4-Infile1'].update('loaded from: ' + fn)
            except:
                window['T4-Infile1'].update('no file loaded')
                work4.clear()
                pass
        elif event == 'T4-Load2':
            try:
                fn = csv_upload_dialog()
                zvar = values['T4-Zvar']
                work4.load_test_sample(fn, zvar)
                window['T4-Infile2'].update('loaded from: ' + fn)
            except:
                window['T4-Infile2'].update('no file loaded')
                work4.clear_test_sample()
                work4.clear_zvar()
                pass
        elif event == 'T4-Test':
            try:
                if t4_canvas is not None:
                    t4_canvas.get_tk_widget().forget()
                nresid = work4.compute_resid()
                t4_figure = work4.resid_plot(nresid)
                if t4_figure is None:
                    raise RuntimeError()
                t4_canvas = draw_figure(window['T4-Canvas'].TKCanvas, t4_figure)
                swres, qqres = work4.resid_test(nresid)
                swstr = f'SW: pass = {str(swres[0]).lower()}, p-value  = {swres[1]:.3f}'
                qqstr = f'QQ: pass = {str(qqres[0]).lower()}, location = {qqres[1]:.3f}, scale = {qqres[2]:.3f}'
                window['T4-Result'].update(swstr + '\n' + qqstr)
            except:
                if t4_figure is not None:
                    t4_figure.clear()
                    t4_canvas = draw_figure(window['T4-Canvas'].TKCanvas, t4_figure)
                window['T4-Result'].update('')
                pass
        # Tab 5
        elif event == 'T5-Load':
            try:
                fn = json_upload_dialog()
                work5.load_model(filename=fn)
                window['T5-Infile'].update('loaded from: ' + fn)
            except:
                window['T5-Infile'].update('no file loaded')
                work5.clear()
                work5.init_critsample()
                pass
        elif event == 'T5-Search':
            try:
                niter = int(values['T5-Niter'])
                niter = max(min(niter, 16), 0)
                work5.explore(niter=niter, show=False, save_to=None)
                window['T5-Table'].update(values=work5.data['critsample'].data.values.tolist())
            except:
                window['T5-Table'].update(values=[])
                pass
        elif event == 'T5-Save':
            try:
                fn = values['T5-Filename']
                work5.save_sample(fn)
                window['T5-Outfile'].update('saved to: ' + fn)
            except:
                window['T5-Outfile'].update('error saving file')
                pass
        elif event == sg.WIN_CLOSED or event == 'Close':
            break

    window.close()
    return False

# returns the current gui scaling
def get_scaling():
    # called before window created
    root = sg.tk.Tk()
    scaling = root.winfo_fpixels('1i')/72
    root.destroy()
    return scaling

# defines the current screen scaling based on the original gui scaling
def def_scaling():
    # original screen parameters when gui was designed.
    or_width, or_height = 2560, 2880   # gotten from sg.Window.get_screen_size()
    # the current screen parameters
    width, height = sg.Window.get_screen_size()
    # the scaling to use
    return get_scaling() * min(width / or_width, height / or_height)

def main():
    sg.set_options(font=('Helvetica', 11))
    sg.theme('LightGrey1')
    sg.set_options(scaling=def_scaling())
    mainwin()

# Executes main
if __name__ == '__main__':
    main()
