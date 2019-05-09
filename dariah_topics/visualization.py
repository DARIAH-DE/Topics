"""
Visualizing the Output of LDA Models
************************************

Functions and classes of this module are for visualizing LDA models. This is, \
except one function (:func:`plot_wordcloud`), based on the document-topics \
distribution DataFrame.

Contents
********
    * :func:`plot_wordcloud` plots the top ``n`` words for a specific topic. \
        The higher their weight, the bigger the word.
    * :clas:`PlotDocumentTopics` is basically the core of this module. Construct \
        this class, if you want to plot the content of the document-topics DataFrame.
    * :meth:`static_heatmap` plots a static, :module:`matplotlib`-based heatmap of \
        the document-topics distribution.
    * :meth:`interactive_heatmap` plots an interactive, :module:`bokeh`-based \
        heatmap of the document-topics distribution.
    * :meth:`static_barchart_per_topic` plots a static, :module:`matplotlib`-based  \
        barchart of the document proportions for a specific topic.
    * :meth:`interactive_barchart_per_topic` plots an interactive, :module:`bokeh`-based  \
        barchart of the document proportions for a specific topic.
    * :meth:`static_barchart_per_document` plots a static, :module:`matplotlib`-based  \
        barchart of the topic proportions for a specific document.
    * :meth:`interactive_barchart_per_document` plots an interactive, :module:`bokeh`-based  \
        barchart of the topic proportions for a specific document.
    * :meth:`topic_over_time` plots a static, :module:`matplotlib`-based  \
        line diagram of the development of a topic over time, based on metadata.
    * :meth:`to_file` saves either a :module:`matplotlib` or a :module:`bokeh` figure \
        object to disk.

"""


import logging
from dariah_topics import postprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import pandas as pd
from bokeh.plotting import figure
from bokeh import palettes
from bokeh.models import (
            ColumnDataSource,
            HoverTool,
            LinearColorMapper,
            BasicTicker,
            ColorBar
            )

from collections import Counter


log = logging.getLogger('dariah_topics')


class PlotDocumentTopics:
    """
    Class to visualize document-topic matrix.
        Args:
            document_topics (pandas.Dataframe): document-topics matrix. Produced by postprocessing.show_document_topics
    """
    def __init__(self, document_topics):
        self.document_topics = document_topics



    def static_heatmap(self, figsize=(1000 / 96, 600 / 96), dpi=None,
                       labels_fontsize=13, cmap='Blues', ticks_fontsize=12,
                       xlabel='Document', ylabel='Topic', xticks_bottom=0.1,
                       xticks_rotation=50, xticks_ha='right', colorbar=False):
        """Plots a static heatmap.
    
        Args:
            figsize (tuple), optional: Size of the figure in inches. Defaults to
                ``(1000 / 96, 500 / 96)``.
            dpi (int), optional: Dots per inch. Defaults to None.
            labels_fontsize (int), optional: Fontsize of the figure labels. Defaults
                to 13.
            cmap (str), optional: Colormap for the figure. Defaults to ``Blues``.
            ticks_fontsize (int), optional: Fontsize of axis ticks. Defaults to 12.
            xlabel (str), optional: Label of x-axis. Defaults to ``Document``.
            ylabel (str), optional: Label of y-axis. Defaults to ``Topic``.
            xticks_bottom (str), optional: Distance to bottom of x-ticks. Defaults
                to 0.1.
            xticks_rotation (int), optional: Rotation degree of x-ticks. Defaults
                to 50.
            xticks_ha (str), optional: The horizontal alignment of the x-tick labels.
                Defaulst to ``right``.
            colorbar (bool), optional: If True, include colorbar. Defaults to True.

        Returns:
            Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        heatmap = ax.pcolor(self.document_topics, cmap=cmap)
        ax.set_xlabel(xlabel, fontsize=labels_fontsize)
        ax.set_ylabel(ylabel, fontsize=labels_fontsize)
        ax.set_xticks(np.arange(self.document_topics.shape[1]) + 0.5)
        ax.set_yticks(np.arange(self.document_topics.shape[0]) + 0.5)
        ax.set_xticklabels(list(self.document_topics.columns), fontsize=ticks_fontsize)
        ax.set_yticklabels(list(self.document_topics.index), fontsize=ticks_fontsize)
        fig.autofmt_xdate(bottom=xticks_bottom, rotation=xticks_rotation, ha=xticks_ha)
        if colorbar:
            cax = ax.imshow(self.document_topics, interpolation='nearest', cmap=cmap)
            cbar = fig.colorbar(cax, ticks=np.arange(0, 1, 0.1))
        return fig
        
    def __static_barchart(self, index, describer, figsize=(11, 7), color='#053967',
                          edgecolor=None, linewidth=None, alpha=None, labels_fontsize=15,
                          ticks_fontsize=14, title=True, title_fontsize=17,
                          dpi=None, transpose_data=False):
        """Plots a static barchart.
    
        Args:
            index Union(int, str): Index of document-topics matrix column or
                name of column.
            describer (str): Describer of what the plot shows, e.g. either document
                or topic.
            title (bool), optional: If True, figure will have a title in the format
                ``describer: index``.
            title_fontsize (int), optional: Fontsize of figure title.
            transpose_data (bool): If True. document-topics matrix will be transposed.
                Defaults to False.
            color (str), optional: Color of the bins. Defaults to ``#053967``.
            edgecolor (str), optional: Color of the bin edges. Defaults to None.
            lindewidth (float), optional: Width of bin lines. Defaults to None.
            alpha (float): Alpha value used for blending. Defaults to None.
            figsize (tuple), optional: Size of the figure in inches. Defaults to
                ``(1000 / 96, 500 / 96)``.
            dpi (int), optional: Dots per inch. Defaults to None.
            labels_fontsize (int), optional: Fontsize of the figure labels. Defaults
                to 15.
            ticks_fontsize (int), optional: Fontsize of axis ticks. Defaults to 14.

        Returns:
            Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi) 
        if isinstance(index, int):
            if transpose_data:
                proportions = self.document_topics.T.iloc[index]
            else:
                proportions = self.document_topics.iloc[index]
            if title:
                plot_title = '{0}: {1}'.format(describer, proportions.name)
                ax.set_title(plot_title, fontsize=title_fontsize)
        elif isinstance(index, str):
            if transpose_data:
                proportions = self.document_topics.T.loc[index]
            else:
                proportions = self.document_topics.loc[index]
            if title:
                plot_title = '{}: {}'.format(describer, index)
                ax.set_title(plot_title, fontsize=title_fontsize)
        else:
            raise ValueError("{} must be int or str.".format(index))
        
        y_axis = np.arange(len(proportions))
        x_axis = proportions
        y_ticks_labels = proportions.index
        ax.barh(y_axis, x_axis, color=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha)
        ax.set_xlabel('Proportion', fontsize=labels_fontsize)
        ax.set_ylabel(describer, fontsize=labels_fontsize)
        ax.set_yticks(y_axis)
        ax.set_yticklabels(y_ticks_labels, fontsize=ticks_fontsize)
        ax.tick_params(axis='x', labelsize=ticks_fontsize)
        return fig

    def static_barchart_per_topic(self, **kwargs):
        """Plots a static barchart per topic.
    
        Args:
            index Union(int, str): Index of document-topics matrix column or
                name of column.
            describer (str): Describer of what the plot shows, e.g. either document
                or topic.
            title (bool), optional: If True, figure will have a title in the format
                ``describer: index``.
            title_fontsize (int), optional: Fontsize of figure title.
            transpose_data (bool): If True. document-topics matrix will be transposed.
                Defaults to False.
            color (str), optional: Color of the bins. Defaults to ``#053967``.
            edgecolor (str), optional: Color of the bin edges. Defaults to None.
            lindewidth (float), optional: Width of bin lines. Defaults to None.
            alpha (float): Alpha value used for blending. Defaults to None.
            figsize (tuple), optional: Size of the figure in inches. Defaults to
                ``(1000 / 96, 500 / 96)``.
            dpi (int), optional: Dots per inch. Defaults to None.
            labels_fontsize (int), optional: Fontsize of the figure labels. Defaults
                to 15.
            ticks_fontsize (int), optional: Fontsize of axis ticks. Defaults to 14.

        Returns:
            Figure object.
        """
        return self.__static_barchart(**kwargs)
        
    def static_barchart_per_document(self, **kwargs):
        """Plots a static barchart per document.
    
        Args:
            index Union(int, str): Index of document-topics matrix column or
                name of column.
            describer (str): Describer of what the plot shows, e.g. either document
                or topic.
            title (bool), optional: If True, figure will have a title in the format
                ``describer: index``.
            title_fontsize (int), optional: Fontsize of figure title.
            transpose_data (bool): If True. document-topics matrix will be transposed.
                Defaults to False.
            color (str), optional: Color of the bins. Defaults to ``#053967``.
            edgecolor (str), optional: Color of the bin edges. Defaults to None.
            lindewidth (float), optional: Width of bin lines. Defaults to None.
            alpha (float): Alpha value used for blending. Defaults to None.
            figsize (tuple), optional: Size of the figure in inches. Defaults to
                ``(1000 / 96, 500 / 96)``.
            dpi (int), optional: Dots per inch. Defaults to None.
            labels_fontsize (int), optional: Fontsize of the figure labels. Defaults
                to 15.
            ticks_fontsize (int), optional: Fontsize of axis ticks. Defaults to 14.

        Returns:
            Figure object.
        """
        return self.__static_barchart(transpose_data=True, **kwargs)

    def interactive_heatmap(self, palette=palettes.Blues[9], reverse_palette=True,
                            tools='hover, pan, reset, save, wheel_zoom, zoom_in, zoom_out',
                            width=1000, height=550, x_axis_location='below', toolbar_location='above',
                            sizing_mode='fixed', line_color=None, grid_line_color=None, axis_line_color=None,
                            major_tick_line_color=None, major_label_text_font_size='9pt',
                            major_label_standoff=0, major_label_orientation=3.14/3, colorbar=True):
        """Plots an interactive heatmap.
    
        Args:
            palette (list), optional: A list of color values. Defaults to ``palettes.Blues[9]``.
            reverse_palette (bool), optional: If True, color values of ``palette`` will
                be reversed. Defaults to True.
            tools (str), optional: Tools, which will be includeded. Defaults to ``hover,
                pan, reset, save, wheel_zoom, zoom_in, zoom_out``.
            width (int), optional: Width of the figure. Defaults to 1000.
            height (int), optional: Height of the figure. Defaults to 550.
            x_axis_location (str), optional: Location of the x-axis. Defaults to
                ``below``.
            toolbar_location (str), optional: Location of the toolbar. Defaults to
                ``above``.
            sizing_mode (str), optional: Size fixed or width oriented. Defaults to ``fixed``.
            line_color (str): Color for lines. Defaults to None.
            grid_line_color (str): Color for grid lines. Defaults to None.
            axis_line_color (str): Color for axis lines. Defaults to None.
            major_tick_line_color (str): Color for major tick lines. Defaults to None.
            major_label_text_font_size (str): Font size for major label text. Defaults
                to ``9pt``.
            major_label_standoff (int): Standoff for major labels. Defaults to 0.
            major_label_orientation (float): Orientation for major labels. Defaults
                to ``3.14 / 3``.
            colorbar (bool): If True, colorbar will be included.

        Returns:
            Figure object.
        """        
        if reverse_palette:
            palette = list(reversed(palette))

        x_range = list(self.document_topics.columns)
        y_range = list(self.document_topics.index)
        
        stacked_data = pd.DataFrame(self.document_topics.stack()).reset_index()
        stacked_data.columns = ['Topics', 'Documents', 'Distributions']
        mapper = LinearColorMapper(palette=palette,
                                   low=stacked_data.Distributions.min(),
                                   high=stacked_data.Distributions.max())
        source = ColumnDataSource(stacked_data)
        
        fig = figure(x_range=x_range,
                     y_range=y_range,
                     x_axis_location=x_axis_location,
                     plot_width=width, plot_height=height,
                     tools=tools, toolbar_location=toolbar_location,
                     sizing_mode=sizing_mode)
        fig.rect(x='Documents', y='Topics', source=source, width=1, height=1,
                 fill_color={'field': 'Distributions', 'transform': mapper},
                 line_color=line_color)

        fig.grid.grid_line_color = grid_line_color
        fig.axis.axis_line_color = axis_line_color
        fig.axis.major_tick_line_color = major_tick_line_color
        fig.axis.major_label_text_font_size = major_label_text_font_size
        fig.axis.major_label_standoff = major_label_standoff
        fig.xaxis.major_label_orientation = major_label_orientation
        
        if 'hover' in tools:
            fig.select_one(HoverTool).tooltips = [('x-Axis', '@Documents'),
                                                  ('y-Axis', '@Topics'),
                                                  ('Score', '@Distributions')]

        if colorbar:
            feature = ColorBar(color_mapper=mapper, major_label_text_font_size=major_label_text_font_size,
                               ticker=BasicTicker(desired_num_ticks=len(palette)),
                               label_standoff=6, border_line_color=None, location=(0, 0))
            fig.add_layout(feature, 'right')
        return fig

    
    def __interactive_barchart(self, index, describer, tools='hover, pan, reset, save, wheel_zoom, zoom_in, zoom_out',
                                width=1000, height=400, toolbar_location='above',
                                sizing_mode='fixed', line_color=None, grid_line_color=None, axis_line_color=None,
                                major_tick_line_color=None, major_label_text_font_size='9pt',
                                major_label_standoff=0, title=True, bin_height=0.5,
                                transpose_data=False, bar_color='#053967'):
        """Plots an interactive barchart.
    
        Args:
            index Union(int, str): Index of document-topics matrix column or
                name of column.
            describer (str): Describer of what the plot shows, e.g. either document
                or topic.
            bar_color (str), optional: Color of bars. Defaults to ``#053967``.
            transpose_data (bool): If True. document-topics matrix will be transposed.
                Defaults to False.
            title (bool), optional: If True, figure will have a title in the format
                ``describer: index``.
            tools (str), optional: Tools, which will be includeded. Defaults to ``hover,
                pan, reset, save, wheel_zoom, zoom_in, zoom_out``.
            width (int), optional: Width of the figure. Defaults to 1000.
            height (int), optional: Height of the figure. Defaults to 400.
            x_axis_location (str), optional: Location of the x-axis. Defaults to
                ``below``.
            toolbar_location (str), optional: Location of the toolbar. Defaults to
                ``above``.
            sizing_mode (str), optional: Size fixed or width oriented. Defaults to ``fixed``.
            line_color (str): Color for lines. Defaults to None.
            grid_line_color (str): Color for grid lines. Defaults to None.
            axis_line_color (str): Color for axis lines. Defaults to None.
            major_tick_line_color (str): Color for major tick lines. Defaults to None.
            major_label_text_font_size (str): Font size for major label text. Defaults
                to ``9pt``.
            major_label_standoff (int): Standoff for major labels. Defaults to 0.

        Returns:
            Figure object.
        """
        if isinstance(index, int):
            if transpose_data:
                proportions = self.document_topics.T.iloc[index]
            else:
                proportions = self.document_topics.iloc[index]
            if title:
                plot_title = '{}: {}'.format(describer, proportions.name)
        elif isinstance(index, str):
            if transpose_data:
                proportions = self.document_topics.T.loc[index]
            else:
                proportions = self.document_topics.loc[index]
            if title:
                plot_title = '{}: {}'.format(describer, index)
        else:
            raise ValueError("{} must be int or str.".format(index))

        x_axis = proportions
        y_range = list(proportions.index)

        source = ColumnDataSource(dict(Describer=y_range, Proportion=x_axis))

        fig = figure(y_range=y_range, title=plot_title, plot_width=width, plot_height=height,
                   tools=tools, toolbar_location=toolbar_location,
                   sizing_mode=sizing_mode)
        fig.hbar(y='Describer', right='Proportion', height=bin_height, source=source,
               line_color=line_color, color=bar_color)

        fig.xgrid.grid_line_color = None
        fig.x_range.start = 0
        fig.grid.grid_line_color = grid_line_color
        fig.axis.axis_line_color = axis_line_color
        fig.axis.major_tick_line_color = major_tick_line_color
        fig.axis.major_label_text_font_size = major_label_text_font_size
        fig.axis.major_label_standoff = major_label_standoff
        
        if 'hover' in tools:
            fig.select_one(HoverTool).tooltips = [('Proportion', '@Proportion')]
        return fig    
    
    def interactive_barchart_per_topic(self, **kwargs):
        """Plots an interactive barchart per topic.
    
        Args:
            index Union(int, str): Index of document-topics matrix column or
                name of column.
            describer (str): Describer of what the plot shows, e.g. either document
                or topic.
            bar_color (str), optional: Color of bars. Defaults to ``#053967``.
            transpose_data (bool): If True. document-topics matrix will be transposed.
                Defaults to False.
            title (bool), optional: If True, figure will have a title in the format
                ``describer: index``.
            tools (str), optional: Tools, which will be includeded. Defaults to ``hover,
                pan, reset, save, wheel_zoom, zoom_in, zoom_out``.
            width (int), optional: Width of the figure. Defaults to 1000.
            height (int), optional: Height of the figure. Defaults to 400.
            x_axis_location (str), optional: Location of the x-axis. Defaults to
                ``below``.
            toolbar_location (str), optional: Location of the toolbar. Defaults to
                ``above``.
            sizing_mode (str), optional: Size fixed or width oriented. Defaults to ``fixed``.
            line_color (str): Color for lines. Defaults to None.
            grid_line_color (str): Color for grid lines. Defaults to None.
            axis_line_color (str): Color for axis lines. Defaults to None.
            major_tick_line_color (str): Color for major tick lines. Defaults to None.
            major_label_text_font_size (str): Font size for major label text. Defaults
                to ``9pt``.
            major_label_standoff (int): Standoff for major labels. Defaults to 0.

        Returns:
            Figure object.
        """
        return self.__interactive_barchart(**kwargs)

    def interactive_barchart_per_document(self, **kwargs):
        """Plots an interactive barchart per document.
    
        Args:
            index Union(int, str): Index of document-topics matrix column or
                name of column.
            describer (str): Describer of what the plot shows, e.g. either document
                or topic.
            bar_color (str), optional: Color of bars. Defaults to ``#053967``.
            transpose_data (bool): If True. document-topics matrix will be transposed.
                Defaults to False.
            title (bool), optional: If True, figure will have a title in the format
                ``describer: index``.
            tools (str), optional: Tools, which will be includeded. Defaults to ``hover,
                pan, reset, save, wheel_zoom, zoom_in, zoom_out``.
            width (int), optional: Width of the figure. Defaults to 1000.
            height (int), optional: Height of the figure. Defaults to 400.
            x_axis_location (str), optional: Location of the x-axis. Defaults to
                ``below``.
            toolbar_location (str), optional: Location of the toolbar. Defaults to
                ``above``.
            sizing_mode (str), optional: Size fixed or width oriented. Defaults to ``fixed``.
            line_color (str): Color for lines. Defaults to None.
            grid_line_color (str): Color for grid lines. Defaults to None.
            axis_line_color (str): Color for axis lines. Defaults to None.
            major_tick_line_color (str): Color for major tick lines. Defaults to None.
            major_label_text_font_size (str): Font size for major label text. Defaults
                to ``9pt``.
            major_label_standoff (int): Standoff for major labels. Defaults to 0.

        Returns:
            Figure object.
        """
        return self.__interactive_barchart(transpose_data=True, **kwargs)

    def topic_over_time(self, metadata_df, threshold=0.1, starttime=1841, endtime=1920):
        """Creates a visualization that shows topics over time.

        Description:
            With this function you can plot topics over time using metadata stored in the documents name.
            Only works with mallet output.

        Args:
            metadata_df(pd.Dataframe()): metadata created by metadata_toolbox
            threshold(float): threshold set to define if a topic in a document is viable
            starttime(int): sets starting point for visualization
            endtime(int): sets ending point for visualization


        Returns: 
            matplotlib plot

        Note: this function is created for a corpus with filenames that looks like:
                1866_ArticleName.txt

        ToDo: make it compatible with gensim output
                Doctest

        """
        years = list(range(starttime, endtime))

        for topiclabel in self.document_topics.index.values:
            topic_over_threshold_per_year = []
            mask = self.document_topics.loc[topiclabel] > threshold
            df = self.document_topics.loc[topiclabel].loc[mask]
            cnt = Counter()
            for filtered_topiclabel in df.index.values:
                year = metadata_df.loc[filtered_topiclabel, 'year']
                print(year)
                cnt[year] += 1
            for year in years:
                topic_over_threshold_per_year.append(cnt[str(year)])
            plt.plot(years, topic_over_threshold_per_year, label=topiclabel)

        plt.xlabel('Year')
        plt.ylabel('count topics over threshold')
        plt.legend()
        # fig.set_size_inches(18.5, 10.5)
        # fig = plt.figure(figsize=(18, 16))
        return plt.gcf().set_size_inches(18.5, 10.5)

    @staticmethod
    def to_file(fig, filename):
        """Saves a figure object to file.
    
        Args:
            fig Union(bokeh.figure, matplotlib.figure): Figure produced by either
                bokeh or matplotlib.
            filename (str): Name of the file with an extension, e.g. ``plot.png``.

        Returns:
            None.
        """
        import matplotlib
        import bokeh
        if isinstance(fig, bokeh.plotting.figure.Figure):
            ext = os.path.splitext(filename)[1]
            if ext == '.png':
                export_png(fig, filename)
            elif ext == '.svg':
                fig.output_backend = 'svg'
                export_svgs(fig, filename)
            elif ext == '.html':
                output_file(filename)
        elif isinstance(fig, matplotlib.figure.Figure):
            fig.savefig(filename)
        return None

