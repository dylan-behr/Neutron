import numpy as np
import re
import csv
import pandas as pd
from scipy.special import erfc, exp1
from scipy.integrate import simps
from math import log10, floor

# Function for rounding to n decimal places
round_to_n = lambda x, n: x if x == 0 else round(x, -int(floor(log10(abs(x)))) + (n - 1))

# Function for finding (-ve) decimal exponent of value
dp = lambda x: 0 if x in [0,0.0] else int(floor(-log10(x%1))+1)

def roots(a,b,c):
    #a,b,c = np.asarray(a), np.asarray(b), np.asarray(c)
    dis = b**2 - 4*a*c
    #if dis >= 0:
    r1 = (-b + np.sqrt(dis))/2*a
    #r2 = (-b - np.sqrt(dis))/2*a
    #     return r1, r2
    # else:
    #     pass
    return r1

def linroot(b,c):
    return -c/b

def split_neg(hkl):
    h_k_l = hkl.split()
    if len(h_k_l) < 3:
        pos = 0
        for k in h_k_l:
            if '-' in k[1:]:
                h_k_l.remove(k)
                new = []
                split = k.split('-')
                if split[0] != '':
                    new.append(split[0])
                for j in split[1:]:
                    new.append('-'+j)
                if pos == 0:
                    new.append(h_k_l[0])
                    h_k_l = new
                else:
                    h_k_l.append(new[0])
                    h_k_l.append(new[1])
            pos += 1
    return h_k_l



def find_index(lines, expression, low = 0, up = 0):
    '''
    Find index of first occurence of expression in line list from open file
    Input:
        lines: data as list of lines; list of str
        expression: expression to be searched for; str
        low, up: lower and upper bounds of line indices over which to search for expression
    Output:
        line index; int
    '''
    if up == 0:
        up = len(lines)
    return np.min([lines[low:up].index(x) + low for x in lines[low:up] if re.search(expression,x)]) 



def extract_prf(path_to_file):

    with open(path_to_file, 'r') as data:
        csv_data = csv.reader(data)
        data_lines = list(csv_data)
        #read in zero, dtt1 and dtt2 for later d-space/TOF conversion
        coef = [float(x) for x in data_lines[1][0].split()[4:7]]
        #find header position
        header = np.min([data_lines.index(x) for x in data_lines if re.search('T.O.F.',x[0])])
        #find the start of hkl reflection data
        startphasedata = np.min([data_lines.index(x) for x in data_lines if re.search('\t        \t        \t        \t',x[0])])
        #length of hkl reflection data
        lenphasedata = len(data_lines) - startphasedata
        #extract TOF, intensity observed, calculated, residuals and background
        spectrumdata = np.genfromtxt(path_to_file, skip_header= header + 1, skip_footer = lenphasedata, delimiter = '\t')
        # extract hkl positions in TOF space and corresponding phase (in multiples of 16(?)) 
        phasedata = np.genfromtxt(path_to_file, skip_header= startphasedata, delimiter= '\t')

        hkldata = []
        for line in data_lines[startphasedata:startphasedata + lenphasedata]:
            line = line[0].split('\t')[-2]
            arr = np.array([int(x) for x in split_neg(line[2:-1])])
            hkldata.append(arr)
        hkldata = np.array(hkldata)

    return spectrumdata, phasedata, coef, hkldata


class Experiment_prf():

    def __init__(self,path_to_file):
        #Attributes
        spectrumdata, phasedata, [zero, dtt1, dtt2], hkldata = extract_prf(path_to_file)
        self.tof  = spectrumdata[:,0]
        self.obs  = spectrumdata[:,1]
        self.calc  = spectrumdata[:,2]
        self.rsds = spectrumdata[:,3]
        self.bgrd = spectrumdata[:,4] 
        self.scale = max(self.obs)*1.05   
        self.ptof = phasedata[:,0]
        phase_labels = sorted(list(set(phasedata[:,5])), reverse= True)
        phase_tof = []
        phase_hkl = []
        phases = []
        phase_count = 0
        for phase in phase_labels:
            phase_tof.append(phasedata[:,0][phasedata[:,5] == phase])
            phase_hkl.append(hkldata[phasedata[:,5] == phase])
            phases.append(phase_count*np.ones(len(phasedata[:,0][phasedata[:,5] == phase])))
            phase_count += 1
        self.phase_tof = phase_tof
        self.phases = phases
        self.phase_hkl = phase_hkl
        self.coef = [zero, dtt1, dtt2]   

        #convert TOF to d-space using quadratic formula
        self.d = np.array(list(map(lambda x: roots(dtt2,dtt1,zero - x), self.tof)))    
        self.phase_d = [np.array(list(map(lambda x: roots(dtt2,dtt1,zero - x), ptof))) for ptof in self.phase_tof]
        
        #convert TOF to d-space using linear formula
        self.lin_d = np.array(list(map(lambda x: linroot(dtt1,zero - x), self.tof)))    
        self.linphase_d = [np.array(list(map(lambda x: linroot(dtt1,zero - x), ptof))) for ptof in self.phase_tof] 


def extract_sum(path_to_file, phase, lattice = True, atom = False):
    '''
    Method to extract lattice parameters and optionally other phase data from .sum file
    Input:
        path_to_file: ; str
        phase: ;int
        atom: Additionally return fractional atomic coordinates + Biso + Occ + Mult.; bool
    Output:
        latts (,ats): Lattice parameters (, and optionally frac. coords. + Other atom params) + uncertainties as row entries in 2D array; np.ndarray
    '''
    # Open .sum file
    with open(path_to_file, 'r') as data:
        lines = data.readlines()
        # Find key indices
        phaseindex = find_index(lines, ' => Phase No.  '+ str(phase) + ' ')
        # Handling depends on whether or not final phase of file
        try:
            endphaseindex = find_index(lines, '------------------------------------------------------------------------------', low = phaseindex+2)
        except ValueError:
            endphaseindex = find_index(lines, ' ==> GLOBAL PARAMETERS FOR PATTERN', low = phaseindex+2)
        cellindex = find_index(lines, ' => Cell parameters', low = phaseindex, up = endphaseindex)

        if atom:
            # Find key indices for extraction of atomic params
            atindex = find_index(lines, '  Name      x     sx       y     sy       z      sz      B   sB   occ. socc.  Mult', phaseindex, cellindex)
            endatindex = find_index(lines, ' ==> ', atindex, cellindex) - 1
            # Make dictionary for distinct atoms 
            ats = {}
            for atline in lines[atindex + 1:endatindex]:
                # Extract names of atoms and extract atom parameters as string using regular expressions
                patatname = re.compile(r'\b[A-Z].*?\b')
                patxyz = re.compile(r'\d\.\d\d\d\d\d\(...\d\)  \d\.\d\d\d\d\d\(...\d\)  \d\.\d\d\d\d\d\(...\d\) .\d\.\d\d\d\(..\d\) .\d\.\d\d\d\(..\d\)  ...')
                [atname] = [atmatch.group(0) for atmatch in patatname.finditer(atline)]
                [atxyz] = [atmatch.group(0) for atmatch in patxyz.finditer(atline)]
                # convert atom params to list of floats
                xyz_ls = [float(x) for x in re.split('\(|\)', atxyz)]
                # reorganise into arrays of atom params and uncertainties 
                xyz, sxyz = np.array(xyz_ls[0::2]), np.array(xyz_ls[1::2] + [0])*np.array([1e-5,1e-5,1e-5,1e-3,1e-3,1])
                # assign dictionary entry for atom to 2D array of params and uncertainties
                ats[atname] = np.vstack((xyz, sxyz)).T
                            
    # Extract lattice params directly from key indices
    if lattice:
        latts = np.loadtxt(path_to_file, skiprows = cellindex+1, max_rows = 6)

    if lattice and atom:
        return latts, ats
    elif lattice and not atom:
        return latts
    elif atom and not lattice:
        return ats
    else:
        return

class Phase_sum():
    '''
    Object hosting unit cell and atomic parameters extracted from sum file using extract_sum() function
    '''
    # lattice and atoms 
    def __init__(self, path_to_file, phase = 1, lattice = True, atom = False, magnet = False, spher = False):
        self.lattice = self.extract_sum(path_to_file, phase, lattice = True, atom = False)[0]
        if atom:
            self.atoms = self.extract_sum(path_to_file, phase, lattice = False, atom = True)[0]
        if magnet:
            self.moments = self.extract_sum(path_to_file, phase, lattice = False, atom = False, magnet = True, spher = spher)[0]

    def extract_sum(self, path_to_file, phase, lattice = True, atom = False, magnet = False, spher = False):
        '''
        Method to extract lattice parameters and optionally other phase data from .sum file
        Input:
            path_to_file: ; str
            phase: ;int
            atom: Additionally return fractional atomic coordinates + Biso + Occ + Mult.; bool
        Output:
            latts (,ats): Lattice parameters (, and optionally frac. coords. + Other atom params) + uncertainties as row entries in 2D array; np.ndarray
        '''
        # Open .sum file
        results = []
        with open(path_to_file, 'r') as data:
            lines = data.readlines()
            # Find key indices
            phaseindex = find_index(lines, ' => Phase No.  '+ str(phase) + ' ')
            # Handling depends on whether or not final phase of file
            try:
                endphaseindex = find_index(lines, '------------------------------------------------------------------------------', low = phaseindex+2)
            except ValueError:
                endphaseindex = find_index(lines, ' ==> GLOBAL PARAMETERS FOR PATTERN', low = phaseindex+2)
            cellindex = find_index(lines, ' => Cell parameters', low = phaseindex, up = endphaseindex)

            if atom:
                # Find key indices for extraction of atomic params
                atindex = find_index(lines, '  Name      x     sx       y     sy       z      sz      B   sB   occ. socc.  Mult', phaseindex, cellindex)
                endatindex = find_index(lines, ' ==> ', atindex, cellindex) - 1
                # Make dictionary for distinct atoms 
                ats = {}
                for atline in lines[atindex + 1:endatindex]:
                    # Extract names of atoms and extract atom parameters as string using regular expressions
                    patatname = re.compile(r'\b[A-Z].*?\b')
                    patxyz = re.compile(r'\d\.\d\d\d\d\d\(...\d\)  \d\.\d\d\d\d\d\(...\d\)  \d\.\d\d\d\d\d\(...\d\) .\d\.\d\d\d\(..\d\) .\d\.\d\d\d\(..\d\)  ...')
                    [atname] = [atmatch.group(0) for atmatch in patatname.finditer(atline)]
                    [atxyz] = [atmatch.group(0) for atmatch in patxyz.finditer(atline)]
                    # convert atom params to list of floats
                    xyz_ls = [float(x) for x in re.split('\(|\)', atxyz)]
                    # reorganise into arrays of atom params and uncertainties 
                    xyz, sxyz = np.array(xyz_ls[0::2]), np.array(xyz_ls[1::2] + [0])*np.array([1e-5,1e-5,1e-5,1e-3,1e-3,1])
                    # assign dictionary entry for atom to 2D array of params and uncertainties
                    ats[atname] = np.vstack((xyz, sxyz)).T
            
                results.append(ats)

            if magnet:
                # Find key indices for extraction of atomic params
                magindex = find_index(lines, '  Name      Mom     sMo     Phi     sPhi     Tet     sTet     MPhas   sMPhas', phaseindex, cellindex)
                endmagindex = find_index(lines, ' ==> ', magindex, cellindex) - 1
                # Make dictionary for distinct atoms 
                mags = {}
                for magline, magnameline in zip(lines[magindex + 4 - spher:endmagindex:2],lines[magindex + 3:endmagindex:2]):
                    # Extract names of atoms and extract atom parameters as string using regular expressions
                    patmagname = re.compile(r'\b[A-Z].*?\b')
                    # patmmm = re.compile(r'\d\.\d\d\d\(.\d\.\d\d\d\)    \d\.\d\d\d\(.\d\.\d\d\d\)    \d\.\d\d\d\(.\d\.\d\d\d\) ')
                    patmmm = re.compile(r'\-?\d+\.\d\d\d\(.\d\.\d\d\d\)\s+\-?\d+\.\d\d\d\(.\d\.\d\d\d\)\s+\-?\d+\.\d\d\d\(.\d\.\d\d\d')
                    [magname] = [magmatch.group(0) for magmatch in patmagname.finditer(magnameline)]
                    [magmmm] = [magmatch.group(0) for magmatch in patmmm.finditer(magline)]
                    # convert atom params to list of floats

                    mmm_ls = [float(x) for x in re.split('\(|\)', magmmm)]
                    # reorganise into arrays of atom params and uncertainties 
                    mmm, smmm = np.array(mmm_ls[0::2]), np.array(mmm_ls[1::2])
                    # assign dictionary entry for atom to 2D array of params and uncertainties
                    mags[magname] = np.vstack((mmm, smmm)).T

                results.append(mags)

        # Extract lattice params directly from key indices
        if lattice:
            latts = np.loadtxt(path_to_file, skiprows = cellindex+1, max_rows = 6)

            results.insert(0,latts)


        return tuple(results)
    
    def a_tabulate(self):
        '''
        Method for printing latex-style table of atom parameters with values in scientific format
        '''
        ats = self.atoms
        for atom in ats:
            st = atom 
            for (x,sx) in zip(ats[atom][:4,0],ats[atom][:4,1]):
                if sx in [0,0.0]:
                    sxround = ''
                    decpl = len(str(x)) - 2
                else:
                    sxround =  '(' + str(round_to_n(sx,1))[-1] + ')'
                    decpl = dp(sx)
                xround = str(round(x,decpl))
                if len(xround)-2 < dp(sx):
                    xround += '0'*(dp(sx) - len(xround) + 2)

                st +=  ' & ' + xround+ sxround

            print(st  + r'\\')

    def m_tabulate(self):
        '''
        Method for printing latex-style table of mag moment parameters with values in scientific format
        '''
        moms = self.moments
        for mom in moms:
            st = mom 
            for (x,sx) in zip(moms[mom][:3,0],moms[mom][:3,1]):
                if sx in [0,0.0]:
                    sxround = ''
                    decpl = len(str(x)) - 2
                else:
                    sxround =  '(' + str(round_to_n(sx,1))[-1] + ')'
                    decpl = dp(sx)
                xround = str(round(x,decpl))
                if len(xround)-2 < dp(sx):
                    xround += '0'*(dp(sx) - len(xround) + 2)

                st +=  ' & ' + xround+ sxround

            print(st  + r'\\')


# define function to turn hkl arrays into printable strings
def hkl_string(arr):
    str_hkl = '{'
    for k in arr:
        if k < 0:
            new_k = r'$\bar{' + str(-k) + '}$'
        else:
            new_k = str(k)
        str_hkl += new_k 

    str_hkl += '}'
    return str_hkl

# plots calculated/observed intensities/hkl reflections from Experiment_prf object 
def prf_plot(exp, ax, limx = [0,0], limy = [0,0], phases = [], tof = False, scale = 1,  obs = True, calc = True, bias = False, bias_adjust = 0, bias_col = 'b', zero_adjust = 0,
 labelo = None, labelc = None, obs_col = 'r', calc_col = 'k', linewidtho = 0, linewidthc = 1, linestyle = '-', marker = '.', markersize = 5, mfc = 'None', ticks = True, 
 ticksize = 50, tickwidth = 10, tick_adjust = 0, tick_sep = 15, xline = False, xlinewidth = 0.5, hkl = False, hkl_adjust = 0, hkl_size = 10, fill = False, base = 0, alpha = 0.5):
    #T.O.F or d-spacing
    if tof:
        x = exp.tof
    else:
        x = exp.lin_d

    cond = True*np.arange(len(x))
    # set limits of plot
    if limx != [0,0]:
        cond = ((x >= limx[0]) & (x < limx[1]))
        ax.set_xlim(limx[0],limx[1])
    if limy != [0,0]:
        ax.set_ylim(limy[0],limy[1])

    #plot observed or calculated intensity, scaled and background adjusted
    if obs:
        # intensity may be filled if chosen
        if fill:
            ax.fill_between(x[cond],(exp.obs - exp.bgrd)[cond]/scale + zero_adjust, base, color = obs_col, alpha = alpha, label = labelo)
        else:    
            ax.plot(x[cond],(exp.obs - exp.bgrd)[cond]/scale + zero_adjust, color = obs_col, marker = marker, markersize = markersize, markerfacecolor= mfc, linewidth = linewidtho, linestyle = linestyle, label = labelo)
    if calc:
        if fill:
            ax.fill_between(x[cond],(exp.calc - exp.bgrd)[cond]/scale + zero_adjust, base, color = calc_col, alpha = alpha, label = labelc)
        else:    
            ax.plot(x[cond],(exp.calc - exp.bgrd)[cond]/scale + zero_adjust, color = calc_col, linewidth = linewidthc, linestyle = linestyle, label = labelc)
    #Plot bias if desired
    if bias:
        ax.plot(x[cond],(exp.obs - exp.calc)[cond]/scale +(-3.5 + bias_adjust)*15/(scale), color = bias_col, linewidth = linewidthc, linestyle = linestyle)

    
    # add phase data
    for i in phases:
        if tof:
            phase_x = exp.phase_tof
        else:
            phase_x = exp.linphase_d
        # subset phase data to those within d-space limits
        tick_x = phase_x[i][phase_x[i] > limx[0]][phase_x[i][phase_x[i] > limx[0]] < limx[1]]
        tick_hkl = exp.phase_hkl[i][phase_x[i] > limx[0]][phase_x[i][phase_x[i] > limx[0]] < limx[1]]
        tick_phase = exp.phases[i][phase_x[i] > limx[0]][phase_x[i][phase_x[i] > limx[0]] < limx[1]]

        # remove all but one hkl/d value from each multiplet
        if tick_x.shape[0] > 0:
            new_tick_x = [tick_x[0]]
            new_tick_hkl = [tick_hkl[0]]
            new_tick_phase = [tick_phase[0]]
            for i in range(1,len(tick_x)):
                if tick_x[i] == tick_x[i-1]:
                    pass
                else:
                    new_tick_x.append(tick_x[i])
                    new_tick_hkl.append(tick_hkl[i])
                    new_tick_phase.append(tick_phase[i])
            tick_x = np.array(new_tick_x)
            tick_hkl = np.array(new_tick_hkl)
            tick_phase = np.array(new_tick_phase)

            # add ticks below plot for reflections if desired
            if ticks:
                ax.scatter(tick_x, (tick_phase-2.5 + tick_adjust)*tick_sep/(scale), marker = '|', s = ticksize, linewidths = tickwidth, color = 'g')
            # add vertical lines to plot for reflections if desired
            if xline:
                for i in range(tick_x.shape[0]):
                    ax.axvline(x= tick_x[i] , color = 'k', linestyle = '--', linewidth = xlinewidth)
            # label reflections by hkl values along top border of plot if desired       
            if hkl:
                # decide what side of tickline to put reflection label
                for i in range(tick_x.shape[0]):
                    if i == 0:
                        ha = 'right'
                    elif i == tick_x.shape[0] -1:
                        ha = 'left'
                    elif abs(tick_x[i] - tick_x[i-1]) > abs(tick_x[i] - tick_x[i-1]):
                        ha = 'right'
                    else:
                        ha = 'left'
                    # annotate reflection family label
                    ax.annotate(hkl_string(tick_hkl[i]), (tick_x[i], 0.95*limy[1] + hkl_adjust), horizontalalignment = ha, size = hkl_size)


#turn string of numbers into list of these numbers, even if - signs filling in for delimiters, used in pcr_print()
def separate_nums(line):
    split_line = line.split()
    new_split_line = []
    for num in split_line:
        if '-' in num[1:]:
            num1 = num.split('-')[0]
            num2 = '-' + num.split('-')[1]
            new_split_line.append(num1)
            new_split_line.append(num2)
        else:
            new_split_line.append(num)
    return new_split_line

#transform vector from one basis to another using new basis vectors of in old basis
def basis_transform(vector, basis):
    alpha = np.linalg.inv(np.array(basis))
    return np.matmul(vector,alpha)

#transform frac coords and other params of pcr, printing transformed params in pcr file format and returning dictionary and list of transformed coords and latt. params respectively  
def pcr_print(path_to_file, phase, trans = [0,0,0], basis = [[1,0,0],[0,1,0],[0,0,1]], to_mag = False, mom_p = [0.0,0.0,0.0], to_str = False, bound = True, zero = False, reset_biso = False, reset_occ = False):

    #read data lines as list of lists of string
    with open(path_to_file, 'r') as data:
        lines = data.readlines()

    #find key line indices bounding data to be extracted
    phaseindex = np.min([lines.index(x) for x in lines if re.search(phase, x)])
    endphaseindex = np.min([lines[phaseindex:].index(x) + phaseindex for x in lines[phaseindex:] if re.search('ABS: ABSCOR1  ABSCOR2', x)])
    jbt = int(lines[phaseindex + 3].split()[3])
    magnetic = (jbt in [1,-1])
    #different key phrases indicating start of frac. coord. data depending on nuclear vs magnetic structure
    if magnetic:
        startatcoord = np.min([lines[phaseindex:endphaseindex].index(x) + phaseindex + 1 for x in lines[phaseindex:endphaseindex] if re.search('!Atom   Typ  Mag Vek    X      Y      Z       Biso    Occ', x)])
        #magnetic data also has 2 extra columns for magnetic rotation matrix and propagation vector identificators that are skipped over 
        start_col = 4
    else:    
        startatcoord = np.min([lines[phaseindex:endphaseindex].index(x) + phaseindex for x in lines[phaseindex:endphaseindex] if re.search('!Atom   Typ       X        Y        Z     Biso', x)])
        start_col = 2
    endatcoord = np.min([lines[startatcoord:].index(x) + startatcoord for x in lines[phaseindex:endphaseindex] if re.search('!-------> Profile Parameters for Pattern #   1', x)])
    latparamindex = np.min([lines[endatcoord:endphaseindex].index(x) + endatcoord for x in lines[endatcoord:endphaseindex] if re.search('!     a          b         c        alpha      beta       gamma      #Cell Info', x)]) + 1
    # lattice parameters extracted
    latparams = [float(x) for x in separate_nums(lines[latparamindex])]
    [a,b,c,alpha,beta,gamma] = latparams
    alpha, beta, gamma = (np.pi/180)*alpha, (np.pi/180)*beta, (np.pi/180)*gamma
    metric = np.array([[a**2, a*b*np.cos(gamma),a*c*np.cos(beta)],[a*b*np.cos(gamma),b**2,b*c*np.cos(alpha)],[a*c*np.cos(beta),b*c*np.cos(alpha),c**2]])
    # x, y, z = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
    # compute new a, b and c from sqrt(a' M a'T) in old basis
    [a,b,c] = [np.sqrt(np.matmul(np.array([a_i]), np.matmul(metric, np.array([a_i]).T))[0][0]) for a_i in basis]
    # compute new alpha, beta and gamma from alpha = cos(b.c/|b||c|) using metric for dot product, all in old basis
    alpha = np.arccos((np.matmul(np.array([basis[1]]), np.matmul(metric, np.array([basis[2]]).T))[0][0])/(b*c))*(180/np.pi)
    beta = np.arccos((np.matmul(np.array([basis[0]]), np.matmul(metric, np.array([basis[2]]).T))[0][0])/(a*c))*(180/np.pi)
    gamma = np.arccos((np.matmul(np.array([basis[0]]), np.matmul(metric, np.array([basis[1]]).T))[0][0])/(a*b))*(180/np.pi)
    newlatparams = [x.round(6) for x in [a,b,c,alpha,beta,gamma]]

    #create dictionary of original atom labels with frac. coords
    atoms = {}
    for line in lines[startatcoord+1:endatcoord:start_col]:
        atoms[line.split()[0]] = np.array([float(x) for x in separate_nums(line)[start_col:start_col + 3]])

    #transform and translate atoms to new coords
    newatoms = {}
    for atom in atoms:
        newatoms[atom] = basis_transform(atoms[atom], basis) + np.array(trans)
        #'bound' keyword transforms coords back to single unit cell by lattice vector translation
        if bound:
            for i in range(len(newatoms[atom])):
                if newatoms[atom][i] < 0:
                    newatoms[atom][i] += 1
                if newatoms[atom][i] > 1:
                    newatoms[atom][i] -= 1

    newlines = []
    for atom in atoms:
        lineindex = np.min([lines[startatcoord:].index(x) + startatcoord for x in lines[startatcoord:endatcoord] if re.search(atom, x)])
        #create list of numbers for frac coord data from lines
        line = separate_nums(lines[lineindex])
        #insert new frac. coords into old ones position in list
        for i in range(3):
            line[start_col + i] = str(newatoms[atom][i].round(5))
        #reset Bisos and occupation nums if keyword True 
        if reset_biso:
            line[start_col + 3] = '1.00'
        if reset_occ:
            line[start_col + 4] = '1.00'
        #turn list of string numbers back into 1 string and append to list of strings 'newlines'
        glue = '  '
        string = glue.join(line)
        newlines.append(string)

        #'zero' keyword fixes all coords and other variables, otherwise leaving intermediate lines unchanged (appropriate for magnetic or nuclear). 
        #These new lines are appended to list 'newlines'
        if type(zero) == bool:

            mom_params = '  ' + glue.join([str(m) for m in mom_p])

            if (magnetic and not to_str) or (not magnetic and to_mag):
                if (not magnetic and to_mag):
                    newlines[-1] = glue.join(separate_nums(newlines[-1])[:-4]) + mom_params
                if zero:
                    newlines.append('                      0.00    0.00    0.00    0.00     0.00    0.00    0.00    0.00')
                    newlines.append('     0.000   0.000   0.000   0.000   0.000   0.000  0.00000')
                    newlines.append('      0.00    0.00    0.00    0.00    0.00    0.00     0.00')

                else:
                    newlines.append(lines[lineindex + 1][:-2] + (not magnetic and to_mag)*mom_params)
                    if (not magnetic and to_mag):
                        newlines.append('     0.000   0.000   0.000   0.000   0.000   0.000  0.00000')
                        newlines.append('      0.00    0.00    0.00    0.00    0.00    0.00     0.00')
                    else:
                        newlines.append(lines[lineindex + 2][:-2])
                        newlines.append(lines[lineindex + 3][:-2])
            else:
                if (magnetic and to_str):
                    newlines[-1] = glue.join(separate_nums(newlines[-1])[:-3]) + '  0  0  0   0'
                    second_line = glue.join(separate_nums(lines[lineindex + 1][:-2])[:-3])
                else:
                    second_line = lines[lineindex + 1][:-2]

                if zero:
                    newlines.append('                  0.00     0.00     0.00     0.00      0.0')  
                else:
                    newlines.append(second_line)

        #'zero' keyword can also be integer/float to shift all refining labels by a certain number (in order to retain some correlations while removing others)
        else:
            label_line = separate_nums(lines[lineindex + 1]) 
            for i in range(len(label_line)):
                if float(label_line[i]) != 0.00:
                    label_line[i] = str(float(label_line[i]) + np.sign(float(label_line[i]))*zero)

            newlines.append(glue.join(label_line))
            if magnetic:
                newlines.append(lines[lineindex + 2][:-2])
                label_line = separate_nums(lines[lineindex + 3]) 
                for i in range(len(label_line)):
                    if float(label_line[i]) != 0.00:
                        label_line[i] = str(float(label_line[i]) + np.sign(float(label_line[i]))*zero)

                newlines.append(glue.join(label_line))

    newlines.append('\n')
    newlines.append('LATTICE PARAMETERS: ')
    newlines.append(glue.join([str(x) for x in newlatparams]))
    #new lines printed in format of orignal pcr file
    for newline in newlines:
        print(newline)

    #transformed atoms return as dictionary
    return newatoms, newlatparams




# return peak profile parameters extracted from .out file
def extract_out(path_to_file, phase, pattern):
    with open(path_to_file, 'r') as data:
        lines = data.readlines()

    #find key line indices bounding data to be extracted
    patpindex = np.min([lines.index(x) for x in lines if re.search(' =>------->  PATTERN number:  ' + str(pattern), x)])
    zeroindex = np.min([lines[patpindex:].index(x) + patpindex for x in lines[patpindex:] if re.search(' => Zero-point: ', x)])
    dtt1index = np.min([lines[patpindex:].index(x) + patpindex for x in lines[patpindex:] if re.search(' => D-spacing to T.O.F. coefficient dtt1 and code:', x)])
    dtt2index = np.min([lines[patpindex:].index(x) + patpindex for x in lines[patpindex:] if re.search(' => D-spacing to T.O.F. coefficient dtt2 and code:', x)])

    phaseindex = np.min([lines.index(x) for x in lines if re.search(phase, x)])
    patternindex = np.min([lines[phaseindex:].index(x) + phaseindex for x in lines[phaseindex:] if re.search(' =>-------> PROFILE PARAMETERS FOR PATTERN:  ' + str(pattern), x)])
    sfindex = np.min([lines[patternindex:].index(x) + patternindex for x in lines[patternindex:] if re.search(' => Overall scale factor:    ', x)])
    gaussindex = np.min([lines[patternindex:].index(x) + patternindex for x in lines[patternindex:] if re.search(' => T.O.F. Gaussian variances Sig-2, Sig-1, Sig-0:', x)])
    lorentzindex = np.min([lines[patternindex:].index(x) + patternindex for x in lines[patternindex:] if re.search(' => T.O.F. Lorentzian FWHM Gam-2, Gam-1, Gam-0:', x)])
    expindex = np.min([lines[patternindex:].index(x) + patternindex for x in lines[patternindex:] if re.search(' => T.O.F. Peak shape parameter alpha0,beta0,alpha1,beta1/kappa:', x)])
    
    #convert lines to list of float peak profile parameters
    patparams = [[float(x) for x in lines[i].split(':')[1].split()][0] for i in [zeroindex, dtt1index, dtt2index]]
    [sf, gaussparams, lorentzparams, expparams] = [[float(x) for x in lines[i].split(':')[1].split()] for i in [sfindex, gaussindex, lorentzindex, expindex]]
    gaussparams, lorentzparams = gaussparams[::-1], lorentzparams[::-1]
    return patparams, sf, gaussparams, lorentzparams, expparams


# return array of peak profile and d-space around reflection from peak profile parameters: A numerical approximation of the voigt-exponential convolution
def peak_profile(d, patparams, sf, gaussparams, lorentzparams, expparams):
    
    #compute TOF peak position from d
    t0 = patparams[0] + patparams[1]*d + patparams[2]*d**2
    #extract scale factor
    sf = sf[0]

    #compute rising and falling back to back exponential parameters from d-dependence, and normalization factor N
    alp = expparams[0] + expparams[2]/d
    bet = expparams[1] + expparams[3]/d**4  
    N = (alp*bet)/(2*(alp + bet))
    
    #compute gaussian std dev and FWHM from d-dependence
    sigGsqr = gaussparams[0] + gaussparams[1]*d**2 + gaussparams[2]*d**4
    HG = np.sqrt(8*np.log(2)*sigGsqr)
    
    #compute lorentzian FWHM from d-dependence
    gamL = lorentzparams[0] + lorentzparams[1]*d + lorentzparams[2]*d**2
    HL = gamL

    #compute pseudo-Voigt FWHM and variance from that of Gaussian and Lorentzian
    H = (HG**5 + 2.69269*HG**4*HL + 2.42843*HG**3*HL**2 + 4.47163*HG**2*HL**3 + 0.07842*HG*HL**4 + HL**5)**0.2
    sigsqr = H**2/(8*np.log(2))

    #compute fraction Omega2 vs Omega1 (degree of lorentzian character?)
    eta = 1.36603*(HL/H) - 0.47719*(HL/H)**2 + 0.11116*(HL/H)**3
    
    #define function to return peak profile in TOF space from pV FWHM/variance and decay params on TOF array 
    def Omega(t, sigsqr, H, alp, bet):
        u = np.array(0.5*alp*(alp*sigsqr + 2*t))
        v = np.array(0.5*bet*(bet*sigsqr - 2*t))
        y = np.array((alp*sigsqr + t)/np.sqrt(2*sigsqr)) 
        z = np.array((bet*sigsqr - t)/np.sqrt(2*sigsqr))
        p = np.array(alp*t + (alp*H/2)*1j)
        q = np.array(-bet*t + (bet*H/2)*1j)

        Omega1 = (1 - eta)*N*(np.multiply(np.exp(u),erfc(y)) + np.multiply(np.exp(v),erfc(z)))
        Omega2 = (2*N*eta/np.pi)*(np.imag(np.multiply(np.exp(p),exp1(p))) + np.imag(np.multiply(np.exp(q),exp1(q))))

        return Omega1 - Omega2
    
    #create initial profile on TOF range 0 +/- 5 FWHM
    t = np.linspace(- 5*H, 5*H, 100) 
    profile = Omega(t, sigsqr, H, alp, bet)

    #find peak centre and reset TOF range to TOF_0 +/- 5 FWHM
    t_max = t[list(profile).index(max(profile))]
    t += t_max

    #define function to convert TOF -> d-space (linear transformation, dtt2 neglected)
    def d_space(t, zero, dtt1):
        return (t - zero)/dtt1

    #return profile over recentred range (symmetric)
    profile = Omega(t, sigsqr, H, alp, bet)

    d = d_space(t + t0, patparams[0], patparams[1])
    #normalise profile
    profile = profile/simps(profile, t)
    #return peak profile array (scaled with scale factor) and d-space array over which peak appears (converted from TOF refl. position +/- 5 FWHM) 
    return sf*profile, d


#return return array of peak profile and d-space around reflection from .out file and d-space of reflection using extract_out() and peak_profile() functions
def refl_profile(path_to_file, phase, pattern, d):
    patparams, sf, gaussparams, lorentzparams, expparams = extract_out(path_to_file, phase, pattern)
    return peak_profile(d, patparams, sf, gaussparams, lorentzparams, expparams)



def profiles_out(path_to_file, phase, pattern, hkls, drange = [0,7],mag = False, tof = False):
    with open(path_to_file, 'r') as data:
        lines = data.readlines()

    #find key line indices bounding data to be extracted
    patpindex = np.min([lines.index(x) for x in lines if re.search(' =>------->  PATTERN number:  ' + str(pattern), x)])
    zeroindex = np.min([lines[patpindex:].index(x) + patpindex for x in lines[patpindex:] if re.search(' => Zero-point: ', x)])
    dtt1index = np.min([lines[patpindex:].index(x) + patpindex for x in lines[patpindex:] if re.search(' => D-spacing to T.O.F. coefficient dtt1 and code:', x)])
    dtt2index = np.min([lines[patpindex:].index(x) + patpindex for x in lines[patpindex:] if re.search(' => D-spacing to T.O.F. coefficient dtt2 and code:', x)])

    phaseindex = np.min([lines.index(x) for x in lines if re.search(' => Phase  No. ' + str(phase), x)])
    patternindex = np.min([lines[phaseindex:].index(x) + phaseindex for x in lines[phaseindex:] if re.search(' =>-------> PROFILE PARAMETERS FOR PATTERN:  ' + str(pattern), x)])
    sfindex = np.min([lines[patternindex:].index(x) + patternindex for x in lines[patternindex:] if re.search(' => Overall scale factor:    ', x)])
    expindex = np.min([lines[patternindex:].index(x) + patternindex for x in lines[patternindex:] if re.search(' => T.O.F. Peak shape parameter alpha0,beta0,alpha1,beta1/kappa:', x)])
    
    #convert lines to list of float peak profile parameters
    [zero, dtt1, dtt2] = [[float(x) for x in lines[i].split(':')[1].split()][0] for i in [zeroindex, dtt1index, dtt2index]]
    [sf, expparams] = [[float(x) for x in lines[i].split(':')[1].split()] for i in [sfindex, expindex]]
    sf = sf[0]
    [alp0, bet0, alp1, bet1] = expparams 
    [tmin, tmax] = [zero + dtt1*d + dtt2*d**2 for d in drange]
    commontspace = np.linspace(tmin, tmax, 500)
    dspace = (commontspace - zero)/dtt1

    patphindex =  np.min([lines.index(x) for x in lines if re.search(' Pattern#  '+ str(pattern) + ' Phase No.:   ' + str(phase), x)])
    headerindex = np.min([lines[patphindex:].index(x) + patphindex for x in lines[patphindex:] if re.search('   No.  Code     H   K   L ', x)])
    endpatphindex = np.min([lines[patphindex+3:].index(x) + patphindex + 3 for x in lines[patphindex + 3:] if re.search('-----------------------------------------', x)])
    
    df0 = pd.read_csv(path_to_file, header = headerindex, nrows = endpatphindex - headerindex, skip_blank_lines = False, delim_whitespace= True)
    df0.drop(['CORR'],axis = 1 ,inplace = True)
    df0.dropna(inplace = True)
    df0 = df0[df0.ne(df0.columns).any(1)]

    if mag:
        int_cols = 7
    else:
        int_cols = 6

    for col in df0.columns[:int_cols]:
        df0[col] = pd.to_numeric(df0[col], downcast= 'integer')    
    for col in df0.columns[int_cols:]:
        df0[col] = pd.to_numeric(df0[col], downcast= 'float')

    Iprofs = []
    labels = []

    for hkl in hkls:
        
        df = df0[df0['H'] == hkl[0]]
        df = df[df['K'] == hkl[1]]
        df = df[df['L'] == hkl[2]]
        [mult, H, I, eta, d] = [df[x].values[0] for x in ['Mult', 'Hw', 'Icalc', 'ETA', 'd-hkl']]

        t0 = zero + dtt1*d + dtt2*d**2
        alp = alp0 + alp1/d
        bet = bet0 + bet1/d**4  
        N = (alp*bet)/(2*(alp + bet))
        sigsqr = H**2/(8*np.log(2))

        tspace = np.linspace(- 5*H, 5*H, 500) 
        u = np.array(0.5*alp*(alp*sigsqr + 2*tspace))
        v = np.array(0.5*bet*(bet*sigsqr - 2*tspace))
        y = np.array((alp*sigsqr + tspace)/np.sqrt(2*sigsqr)) 
        z = np.array((bet*sigsqr - tspace)/np.sqrt(2*sigsqr))
        p = np.array(alp*tspace + (alp*H/2)*1j)
        q = np.array(-bet*tspace + (bet*H/2)*1j)

        Omega1 = (1 - eta)*N*(np.multiply(np.exp(u),erfc(y)) + np.multiply(np.exp(v),erfc(z)))
        Omega2 = (2*N*eta/np.pi)*(np.imag(np.multiply(np.exp(p),exp1(p))) + np.imag(np.multiply(np.exp(q),exp1(q))))

        Iprof = (Omega1 - Omega2)*I
        tspace += t0
        # tspace[0], tspace[-1] = tmin, tmax
        Iprof = np.interp(commontspace, tspace, Iprof)
        if hkl != hkls[0] and sum(abs(Iprof - Iprofs[-1])) < 0.000001:
            Iprofs[-1] += Iprof
        else:
            Iprofs.append(Iprof)
            labels.append(hkl_string(hkl))
    Iprofs = np.array(Iprofs)
    if tof:
        x = commontspace
    else:
        x = dspace

    return Iprofs, x, labels


# Function to return summary of how many variables are being refined of each number, to check that no unwanted correlations are being modelled
def param_counts(path_to_file):
    with open(path_to_file, 'r') as data:
        lines = data.readlines()
    
    param_totindex = np.min([lines.index(x) for x in lines if re.search('!Number of refined parameters', x)])
    param_tot = int(lines[param_totindex].split('!')[0]) 

    for i in range(1,param_tot+ 1):
        counts = []
        index = 0
        for line in lines:
            index += 1
            if re.search(' ' + str(i) + '1.00', line):
                counts.append(index)
        print(str(i) + '1.00 : ' + str(len(counts)) +' occurences : '+ str(counts))
    return param_tot

# Extract scale factor of phase from .sum file in form [SFavg, SFstddev]
def scale_from_sum(address, phase):
    with open(address, 'r') as data:
        lines = data.readlines()

    #find key line indices bounding data to be extracted
    phaseindex = np.min([lines.index(x) for x in lines if re.search(' => Phase No.  ' + str(phase), x)])
    sfindex = np.min([lines[phaseindex:].index(x) + phaseindex for x in lines[phaseindex:] if re.search(' => Overall scale factor :',x)])
    sf = [float(x) for x in lines[sfindex].split(':')[1].split()]
    return np.array(sf)

# Extract scale factor of phase from .sum file in form [SFmin, SFavg, SFmax]
def scale_from_sum2(address, phase):
    with open(address, 'r') as data:
        lines = data.readlines()

    #find key line indices bounding data to be extracted
    phaseindex = np.min([lines.index(x) for x in lines if re.search(' => Phase No.  ' + str(phase), x)])
    sfindex = np.min([lines[phaseindex:].index(x) + phaseindex for x in lines[phaseindex:] if re.search(' => Overall scale factor :',x)])
    sf = [float(x) for x in lines[sfindex].split(':')[1].split()]
    return np.array([sf[0] - sf [1], sf[0], sf[0] + sf [1]])

# Extract moment of magnetic phase from .sum file 
def moment_from_sum(address, phase, spin, vek = 1, polar = False):
    '''
    Extracts moments and std. errors as array vectors from .sum file
    Inputs:
        address: File address; string
        phase: Phase number; int 
        spin: Name of magnetic moment site; string
        vek: 'th propagation vector for this spin; int
        polar: Returns in polar (as per FP definition) form when True, Cartesian otherwise; bool

    Outputs:
        moment vector; numpy_darray
        moment vector standard errors: numpy_darray
    '''
    with open(address, 'r') as data:
        lines = data.readlines()

    #find key line indices bounding data to be extracted
    phaseindex = np.min([lines.index(x) for x in lines if re.search(' => Phase No.  ' + str(phase), x)])
    magmomindex = np.min([lines[phaseindex:].index(x) + phaseindex for x in lines[phaseindex:] if re.search(' ==> MAGNETIC MOMENT PARAMETERS:',x)])
    spindex = np.min([lines[magmomindex:].index(x) + magmomindex for x in lines[magmomindex:] if re.search(spin,x)]) + 1

    if polar:
        spindex -= 1
    
    # Assumes k-vectors of the same magnetic ion are consecutive 
    spindex += 2*(vek-1)

    momls = [float(x) for x in re.split('\(|\)',lines[spindex][5:-2])][:6]
    mom, err_mom = np.array(momls[::2]), np.array(momls[1::2])
    return mom, err_mom

def angle(v1, v2):
    '''
    Returns angle between two vectors
    Inputs:
        v1, v2: input vectors; numpy_darray
    Outputs:
        Angle in radians; float
    '''
    [v1, v2] = [np.array(v) for v in [v1, v2]]
    [v1n, v2n] = [np.linalg.norm(v) for v in [v1, v2]]
    return np.arccos(np.dot(v1, v2)/(v1n*v2n))