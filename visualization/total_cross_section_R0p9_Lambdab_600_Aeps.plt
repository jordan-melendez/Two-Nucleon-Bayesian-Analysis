# gnuplot script to plot total cross sections with error bands
reset  # start from defaults
clear

# energy
# energy = "200"

# common label for output files: get 1,2,3,4,5 appended
out_file_base = "X0_DOB_0.68-0.95_Lambda-600_C_t-t-t-t_vs_energy-1-351-1"

# input file names with cross section and DoB bands
relative_path = "../observables_with_errors/"
observable_file = "X0_DOB_0.68-0.95_Lambda-600_C_t-t-t-t_vs_energy-1-351-1"
file_LOp = relative_path . observable_file . "_LOp_vnn_kvnn_41.dat"
file_LO = relative_path . observable_file . "_LO_vnn_kvnn_41.dat"
file_NLO = relative_path . observable_file . "_NLO_vnn_kvnn_46.dat"
file_N2LO = relative_path . observable_file . "_N2LO_vnn_kvnn_51.dat"
file_N3LO = relative_path . observable_file . "_N3LO_vnn_kvnn_56.dat"
file_N4LO = relative_path . observable_file . "_N4LO_vnn_kvnn_61.dat"
file_exp = "pwa_C_t-t-t-t.dat"

# Overall title for each of the plots (can omit with title_string = "")
title_string = "{/Symbol s}_{np} for EKM R=0.9 fm, {/Symbol L}_b = 600 MeV, prior set A_{/Symbol e}"

# Monochrome color palates in RGB form
#RGB_1a = "#4e2b4b"  #purple
#RGB_1b = "#8a5689"
#RGB_1c = "#e4dce7"
RGB_1a = "#855a7b"  #purple
RGB_1b = "#bdaeca"
RGB_1c = "#dbdaec"
RGB_2a = "#813919"  # brown
RGB_2b = "#c97b42"
RGB_2c = "#ffdbaf"
# RGB_3a = "#ca1414"  # red
# RGB_3b = "#e9a8a1"
# RGB_3c = "#f4d4d3"
RGB_3a = "#216100"  # green
RGB_3b = "#90b081"
RGB_3c = "#bdd986"
# RGB_4a = "#216100"  # green
# RGB_4b = "#90b081"
# RGB_4c = "#bdd986"
RGB_4a = "#5584b1"  # blue
RGB_4b = "#85c1e5"
RGB_4c = "#cbe2ef"
# RGB_5a = "#5584b1"  # blue
# RGB_5b = "#85c1e5"
# RGB_5c = "#cbe2ef"
RGB_5a = "#ca1414"  # red
RGB_5b = "#e9a8a1"
RGB_5c = "#f4d4d3"

# x and y ranges; set globally here but reset below as needed
xmin = 0
xmax = 350
ymin = 10
ymax = 1000

# linewidths 
lw1 = 3
lw2 = 4

# make fonts bigger for output
set tics font ", 20"
set xlabel font ", 20"
set ylabel font ", 20"

# semi-log plot
set logscale y
# specify the postscript font
ps_font = "Helvetica,16"

# labels
my_x_label = 'E [MeV]'
my_y_label = 'cross section [mb]'

######################

# LO only
plotnum = 1
# set term x11 plotnum
set size square
set key top right

set title title_string
set xrange [xmin:xmax]
set xlabel my_x_label
set yrange [ymin:ymax]
set ylabel my_y_label


# add labels on top
#set label "68% DoB" at 180,57 front
#set label "95% DoB" at 230,120 front

plot \
  file_LO using ($1):(($2)-($4)):(($2)+($4)) with filledcurves lt rgb RGB_1c notitle, \
  file_LO using ($1):(($2)-($3)):(($2)+($3)) with filledcurves lt rgb RGB_1b notitle, \
  file_LO using 1:2 with lines lw lw1 lt rgb RGB_1a title "LO" , \
  file_exp using 1:2 with lines lw lw2 lt rgb "black" title "NPWA" 

set term postscript eps color enhanced font ps_font
out_file = out_file_base . "_" . plotnum . ".ps"
set out out_file
replot
system_string = "bbox_add.pl " . out_file
system system_string
system_string = "epstopdf " . out_file
system system_string

unset label  # turn off the DoB labels

######################

# LO, NLO
plotnum = 2
# set term x11 plotnum
set size square
set key top right

set title title_string
set xrange [xmin:xmax]
set xlabel my_x_label
set yrange [ymin:ymax]
set ylabel my_y_label

plot \
  file_LO using ($1):(($2)-($4)):(($2)+($4)) with filledcurves lt rgb RGB_1c notitle, \
  file_LO using ($1):(($2)-($3)):(($2)+($3)) with filledcurves lt rgb RGB_1b notitle, \
  file_LO using 1:2 with lines lw lw1 lt rgb RGB_1a title "LO" , \
  file_NLO using ($1):(($2)-($4)):(($2)+($4)) with filledcurves lt rgb RGB_2c notitle, \
  file_NLO using ($1):(($2)-($3)):(($2)+($3)) with filledcurves lt rgb RGB_2b notitle, \
  file_NLO using 1:2 with lines lw lw1 lt rgb RGB_2a title "NLO" , \
  file_exp using 1:2 with lines lw lw2 lt rgb "black" title "NPWA" 

set term postscript eps color enhanced
out_file = out_file_base . "_" . plotnum . ".ps"
set out out_file
replot
system_string = "bbox_add.pl " . out_file
system system_string
system_string = "epstopdf " . out_file
system system_string

######################

# LO, NLO, N2LO
plotnum = 3
# set term x11 plotnum
set size square
set key top right

set title title_string
set xrange [xmin:xmax]
set xlabel my_x_label
set yrange [ymin:ymax]
set ylabel my_y_label

plot \
  file_LO using ($1):(($2)-($4)):(($2)+($4)) with filledcurves lt rgb RGB_1c notitle, \
  file_LO using ($1):(($2)-($3)):(($2)+($3)) with filledcurves lt rgb RGB_1b notitle, \
  file_LO using 1:2 with lines lw lw1 lt rgb RGB_1a title "LO" , \
  file_NLO using ($1):(($2)-($4)):(($2)+($4)) with filledcurves lt rgb RGB_2c notitle, \
  file_NLO using ($1):(($2)-($3)):(($2)+($3)) with filledcurves lt rgb RGB_2b notitle, \
  file_NLO using 1:2 with lines lw lw1 lt rgb RGB_2a title "NLO" , \
  file_N2LO using ($1):(($2)-($4)):(($2)+($4)) with filledcurves lt rgb RGB_3c notitle, \
  file_N2LO using ($1):(($2)-($3)):(($2)+($3)) with filledcurves lt rgb RGB_3b notitle, \
  file_N2LO using 1:2 with lines lw lw1 lt rgb RGB_3a title "N2LO" , \
  file_exp using 1:2 with lines lw lw2 lt rgb "black" title "NPWA" 

set term postscript eps color enhanced
out_file = out_file_base . "_" . plotnum . ".ps"
set out out_file
replot
system_string = "bbox_add.pl " . out_file
system system_string
system_string = "epstopdf " . out_file
system system_string

######################

# LO, NLO, N2LO, N3LO
plotnum = 4
# set term x11 plotnum
set size square
set key top right

set title title_string
set xrange [xmin:xmax]
set xlabel my_x_label
set yrange [ymin:ymax]
set ylabel my_y_label

plot \
  file_LO using ($1):(($2)-($4)):(($2)+($4)) with filledcurves lt rgb RGB_1c notitle, \
  file_LO using ($1):(($2)-($3)):(($2)+($3)) with filledcurves lt rgb RGB_1b notitle, \
  file_LO using 1:2 with lines lw lw1 lt rgb RGB_1a title "LO" , \
  file_NLO using ($1):(($2)-($4)):(($2)+($4)) with filledcurves lt rgb RGB_2c notitle, \
  file_NLO using ($1):(($2)-($3)):(($2)+($3)) with filledcurves lt rgb RGB_2b notitle, \
  file_NLO using 1:2 with lines lw lw1 lt rgb RGB_2a title "NLO" , \
  file_N2LO using ($1):(($2)-($4)):(($2)+($4)) with filledcurves lt rgb RGB_3c notitle, \
  file_N2LO using ($1):(($2)-($3)):(($2)+($3)) with filledcurves lt rgb RGB_3b notitle, \
  file_N2LO using 1:2 with lines lw lw1 lt rgb RGB_3a title "N2LO" , \
  file_N3LO using ($1):(($2)-($4)):(($2)+($4)) with filledcurves lt rgb RGB_4c notitle, \
  file_N3LO using ($1):(($2)-($3)):(($2)+($3)) with filledcurves lt rgb RGB_4b notitle, \
  file_N3LO using 1:2 with lines lw lw1 lt rgb RGB_4a title "N3LO" , \
  file_exp using 1:2 with lines lw lw2 lt rgb "black" title "NPWA" 

set term postscript eps color enhanced
out_file = out_file_base . "_" . plotnum . ".ps"
set out out_file
replot
system_string = "bbox_add.pl " . out_file
system system_string
system_string = "epstopdf " . out_file
system system_string

######################


# LO, NLO, N2LO, N3LO, N4LO
plotnum = 5
# set term x11 plotnum
set size square
set key top right

set title title_string
set xrange [xmin:xmax]
set xlabel my_x_label
set yrange [ymin:ymax]
set ylabel my_y_label

plot \
  file_LO using ($1):(($2)-($4)):(($2)+($4)) with filledcurves lt rgb RGB_1c notitle, \
  file_LO using ($1):(($2)-($3)):(($2)+($3)) with filledcurves lt rgb RGB_1b notitle, \
  file_LO using 1:2 with lines lw lw1 lt rgb RGB_1a title "LO" , \
  file_NLO using ($1):(($2)-($4)):(($2)+($4)) with filledcurves lt rgb RGB_2c notitle, \
  file_NLO using ($1):(($2)-($3)):(($2)+($3)) with filledcurves lt rgb RGB_2b notitle, \
  file_NLO using 1:2 with lines lw lw1 lt rgb RGB_2a title "NLO" , \
  file_N2LO using ($1):(($2)-($4)):(($2)+($4)) with filledcurves lt rgb RGB_3c notitle, \
  file_N2LO using ($1):(($2)-($3)):(($2)+($3)) with filledcurves lt rgb RGB_3b notitle, \
  file_N2LO using 1:2 with lines lw lw1 lt rgb RGB_3a title "N2LO" , \
  file_N3LO using ($1):(($2)-($4)):(($2)+($4)) with filledcurves lt rgb RGB_4c notitle, \
  file_N3LO using ($1):(($2)-($3)):(($2)+($3)) with filledcurves lt rgb RGB_4b notitle, \
  file_N3LO using 1:2 with lines lw lw1 lt rgb RGB_4a title "N3LO" , \
  file_N4LO using ($1):(($2)-($4)):(($2)+($4)) with filledcurves lt rgb RGB_5c notitle, \
  file_N4LO using ($1):(($2)-($3)):(($2)+($3)) with filledcurves lt rgb RGB_5b notitle, \
  file_N4LO using 1:2 with lines lw lw1 lt rgb RGB_5a title "N4LO" , \
  file_exp using 1:2 with lines lw lw2 lt rgb "black" title "NPWA" 

set term postscript eps color enhanced
out_file = out_file_base . "_" . plotnum . ".ps"
set out out_file
replot
system_string = "bbox_add.pl " . out_file
system system_string
system_string = "epstopdf " . out_file
system system_string

######################
