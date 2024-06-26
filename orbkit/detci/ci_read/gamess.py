import numpy
from copy import copy

from orbkit.units import ev_to_ha
from orbkit.display import display
from orbkit.qcinfo import CIinfo
from orbkit.read.tools import descriptor_from_file

from .tools import multiplicity

def gamess_tddft(fname,select_state=None,threshold=0.0,**kwargs):
  '''Reads GAMESS-US TDDFT output. 
  
  **Parameters:**
  
    fname: str, file descriptor
      Specifies the filename for the input file.
      fname can also be used with a file descriptor instad of a filename.
    select_state : None or list of int, optional
      If not None, specifies the states to be read (0 corresponds to the ground 
      state), else read all electronic states.
    threshold : float, optional
      Specifies a read threshold for the CI coefficients.
  
  **Returns:**
  
    ci : list of CIinfo class instances
      See :ref:`Central Variables` for details.
  '''
  display('\nReading data of TDDFT calculation from GAMESS-US...')
  # Initialize variables
  ci = []
  ci_flag = False
  prttol = False
  init_state = False
  rhfspin = 0
  spin = 'Unknown'
  
  if isinstance(select_state,int): select_state = [select_state]

  if isinstance(fname, str):
    filename = fname
    fname = descriptor_from_file(filename, index=0, ci_descriptor=True)
  else:
    filename = fname.name

  for line in fname:
    thisline = line.split()             # The current line split into segments
    #--- Check the file for keywords ---
    # Initialize Hartree-Fock ground state
    if 'NUMBER OF ELECTRONS' in line:
      nel = int(thisline[-1])
    elif 'SPIN MULTIPLICITY' in line:
      rhfspin = int(thisline[-1])
    elif 'SINGLET EXCITATIONS' in line:
      spin = 'Singlet'
    #elif ' FINAL RHF ENERGY IS' in line and (select_state is None or 0 in select_state):
    elif ' FINAL' in line and ' ENERGY IS' in line and (select_state is None or 0 in select_state):
        ci.append(CIinfo(method='tddft'))
        ci[-1].info   = []
        ci[-1].coeffs = []
        ci[-1].occ    = []
        ci[-1].occ.append([0,0])
        ci[-1].coeffs.append(1.0)
        ci[-1].info = {'state': '0',
                       'energy': float(thisline[4]),
                       'fileinfo': filename,
                       'read_threshold': threshold,
                       'spin': spin,
                       'nel': nel}
    # Initialize new excited state
    elif 'STATE #' in line and 'ENERGY =' in line:
      if select_state is None or int(thisline[2]) in select_state:
        init_state = True
        tddft_skip = 8
        ci.append(CIinfo(method='cis'))
        ci[-1].info   = []
        ci[-1].coeffs = []
        ci[-1].occ    = []
        ci[-1].info = {'state': thisline[2],
                       'energy': float(thisline[-2])*ev_to_ha + ci[0].info['energy'],
                       'fileinfo': filename,
                       'read_threshold': threshold,
                       'spin': 'Unknown',
                       'nel': nel}
    if init_state == True and line != '\n' and 'WARNING:' not in line:
      if not tddft_skip:
        if 'NON-ABELIAN' in line or 'SUMMARY' in line or 'SYMMETRY' in line or 'STATE #' in line:
          init_state = False
        else:
          if abs(float(thisline[2])) > threshold:
            ci[-1].occ.append(thisline[:2])
            ci[-1].coeffs.append(thisline[2])
      elif tddft_skip:
        tddft_skip -= 1
          
  fname.close()
          
  #--- Calculating norm of CI states
  display('\nIn total, %d states have been read.' % len(ci)) 
  display('Norm of the states:')
  for i in range(len(ci)):
    j = numpy.array(ci[i].coeffs,dtype=float)
    norm = numpy.sum(j**2)
    ci[i].coeffs = j
    # Write Norm to log-file
    display('\tState %s:\tNorm = %0.8f (%d Coefficients)' % (ci[i].info['state'],norm, len(ci[i].coeffs)))
    # Transform to numpy arrays
    ci[i].occ = numpy.array([s for s in ci[i].occ],dtype=numpy.intc)-1
  
  return ci

def gamess_cis(filename,select_state=None,threshold=0.0,**kwargs):
  '''Reads GAMESS-US CIS output. 
  
  **Parameters:**
  
    filename : str
      Specifies the filename for the input file.
    select_state : None or list of int, optional
      If not None, specifies the states to be read (0 corresponds to the ground 
      state), else read all electronic states.
    threshold : float, optional
      Specifies a read threshold for the CI coefficients.
  
  **Returns:**
  
    ci : list of CIinfo class instances
      See :ref:`Central Variables` for details.
  '''
  display('\nReading data of CIS calculation from GAMESS-US...')
  # Initialize variables
  ci = []
  ci_flag = False
  prttol = False
  init_state = False
  rhfspin = 0
  min_c = -1
  
  if isinstance(select_state,int): select_state = [select_state]
  with open(filename) as fileobject:
    for line in fileobject:
      thisline = line.split()             # The current line split into segments
      #--- Check the file for keywords ---
      # Initialize Hartree-Fock ground state
      if 'NUMBER OF ELECTRONS' in line and "=" in line:
        nel = int(thisline[-1])
      elif 'SPIN MULTIPLICITY' in line:
        rhfspin = int(thisline[-1])
      elif ' FINAL RHF ENERGY IS' in line and (select_state is None or 0 in select_state):
          ci.append(CIinfo(method='cis'))
          ci[-1].info   = []
          ci[-1].coeffs = []
          ci[-1].occ    = []
          ci[-1].occ.append([0,0])
          ci[-1].coeffs.append(1.0)
          ci[-1].info = {'state': '0',
                         'energy': float(thisline[4]),
                         'fileinfo': filename,
                         'read_threshold': threshold,
                         'spin': multiplicity()[rhfspin],
                         'nel': nel}
      # Printing parameter
      elif ' PRINTING CIS COEFFICIENTS LARGER THAN' in line:
        min_c = float(thisline[-1])
      # Initialize new excited state
      elif ' EXCITED STATE ' in line and 'ENERGY=' and 'SPACE SYM' in line:
        if select_state is None or int(thisline[2]) in select_state:
          init_state = True
          cis_skip = 6
          ci.append(CIinfo(method='cis'))
          ci[-1].info   = []
          ci[-1].coeffs = []
          ci[-1].occ    = []
          ci[-1].info = {'state': thisline[2],
                         'energy': float(thisline[4]),
                         'fileinfo': filename,
                         'read_threshold': threshold,
                         'spin': multiplicity()[int(2*float(thisline[7])+1)],
                         'nel': nel}
      if init_state == True:
        if not cis_skip:
          if '----------------------------------------------' in line:
            init_state = False
          else:
            if abs(float(thisline[2])) > threshold:
              ci[-1].occ.append(thisline[:2])
              ci[-1].coeffs.append(thisline[2])
        elif cis_skip:
          cis_skip -= 1
          
  #--- Calculating norm of CI states
  display('\nIn total, %d states have been read.' % len(ci)) 
  display('Norm of the states:')
  for i in range(len(ci)):
    j = numpy.array(ci[i].coeffs,dtype=float)
    norm = numpy.sum(j**2)
    ci[i].coeffs = j
    # Write Norm to log-file
    display('\tState %s:\tNorm = %0.8f (%d Coefficients)' % (ci[i].info['state'],norm, len(ci[i].coeffs)))
    # Transform to numpy arrays
    ci[i].occ = numpy.array([s for s in ci[i].occ],dtype=numpy.intc)-1
  display('')
  if min_c > threshold:
    display('\nInfo:'+
       '\n\tSmallest coefficient (|c|=%f) is larger than the read threshold (%f).' 
       %(min_c,threshold) + 
       '\n\tUse `PRTTOL=0.0` in the `$CIS` input card to print ' +
       'all CI coefficients.\n')
  
  return ci

def gamess_guga(file_name,threshold=0.0,**kwargs):
  '''Reads GAMESS-US GUGA output.

  **Parameters:**

    file_name: str, file descriptor
      Specifies the filename for the input file.
      fname can also be used with a file descriptor instad of a filename.
    threshold : float, optional
      Specifies a read threshold for the CI coefficients.

  **Returns:**

    ci : list of CIinfo class instances
      See :ref:`Central Variables` for details.
  '''
  display('\nReading data of GUGA calculation from GAMESS-US...')

  ci = [[]]
  general_information = {'fileinfo': file_name,
                         'read_threshold': threshold}
 

  fname = open(file_name, 'r')
  
  
  from io import TextIOWrapper
  if isinstance(fname, TextIOWrapper):
    flines = fname.readlines()
  
  STATE_ZONE=False
  state_energies = []
  state_CSFs = []
  state_coeffs = []
  state_OCCs = []
  
  NUMCI = 0
  NFZC = []
  NMCC = []
  NDOC = []
  NAOS = []
  NBOS = []
  NALP = []
  NVAL = []
  NEXT = []
  NFZV = []
  nel = []
  spin = []
  
  for i,line in enumerate(flines):
    if STATE_ZONE:
      thisline = line.split()
      if " STATE #" in line:
        state_energies[-1].append(float(thisline[-1]))
        temp_state_CSFs = []
        temp_state_OCCs = []
        temp_state_coeffs = []
        continue
      if len(thisline) == 3 and abs(float(thisline[1])) >= threshold:
        temp_state_CSFs.append(int(thisline[0]))
        temp_state_coeffs.append(float(thisline[1]))
        temp = [int(x) for x in thisline[2]]
        temp_state_OCCs.append(temp)
        continue
      if " THE " in line:
        state_CSFs[-1].append(temp_state_CSFs)
        state_coeffs[-1].append(temp_state_coeffs)
        state_OCCs[-1].append(temp_state_OCCs)
        continue
      if "STEP CPU TIME" in line:
        STATE_ZONE = False
        continue
      
    if "CI CALCULATION" in line:
      NUMCI += 1
      ci.append([])
      state_energies.append([])
      state_CSFs.append([])
      state_coeffs.append([])
      state_OCCs.append([])
      NFZC.append(int(flines[i+14].split()[1]))
      NDOC.append(int(flines[i+14].split()[3]))
      NEXT.append(int(flines[i+14].split()[5]))
      NMCC.append(int(flines[i+15].split()[1]))
      NAOS.append(int(flines[i+15].split()[3]))
      NFZV.append(int(flines[i+15].split()[5]))
      NBOS.append(int(flines[i+16].split()[1]))
      NALP.append(int(flines[i+17].split()[1]))
      NVAL.append(int(flines[i+18].split()[1]))
      nel.append( 2*(NFZC[-1]+NDOC[-1]+NMCC[-1]) + 1*(NAOS[-1]+NBOS[-1]+NALP[-1]))
      spin.append( 1 + NALP[-1] ) 
      continue
          
    if " NUMBER OF STATES REQUESTED =" in line:
      STATE_ZONE=True
      numstates = int(line.split()[-1])
      continue
          
  fname.close()

  NCORE = [sum(i) for i in zip(*[NFZC,NMCC])]
  NACT = [sum(i) for i in zip(*[NDOC,NAOS,NBOS,NALP,NVAL,NEXT])] 
 
  CSFs_l = []
  for stg_CSFs in state_CSFs:
    CSFs = set()
    for st_CSFs in stg_CSFs:
      for c in st_CSFs:
        CSFs.add(c)
    CSFs_l.append(sorted(list(CSFs)))
  
  CSFs = [{}]
  
  fname = open(file_name, 'r')
  
  if isinstance(fname, TextIOWrapper):
    flines = fname.readlines()
      
  CI_index = 0
  SKIP = 0

  for i,line in enumerate(flines):
    if "CI CALCULATION" in line:
      SKIP = 0
    if SKIP:
      continue    
    if  " CASE VECTOR =" in line:
      if int(line.split()[3]) == CSFs_l[CI_index][0]:
        CSFs[CI_index][CSFs_l[CI_index][0]] = [int(x) for x in flines[i+1].split()[0]]
        CSFs_l[CI_index].pop(0)
        if len(CSFs_l[CI_index])==0:
          CI_index += 1
          CSFs.append({})
          SKIP = 1
          continue

  CSFs.pop(-1)

  fname.close()
  
  spinz = []
  for s in spin:
    spinz.append([s-1-2*x for x in range(s)]) 
   
  state_num = 0
  for CI_i in range(NUMCI):
    for i in range(len(state_energies[CI_i])):
      for ms in spinz[CI_i]:
        ci[CI_i].append(CIinfo(method="guga"))
        ci[CI_i][-1].info = copy(general_information)
        ci[CI_i][-1].info["state"] = state_num
        state_num += 1
        ci[CI_i][-1].info["energy"] = state_energies[CI_i][i]
        ci[CI_i][-1].info["spin"] = spin[CI_i]
        ci[CI_i][-1].info["ms"] = ms/2
        ci[CI_i][-1].info["nel"] = nel[CI_i]
        ci[CI_i][-1].info["occ_info"] = [NCORE[CI_i],NACT[CI_i],NFZV[CI_i]]
        for j,csfi in enumerate(state_CSFs[CI_i][i]):
          CGCs, OCCs = CV_to_dets(CSFs[CI_i][csfi], NCORE[CI_i], NDOC[CI_i], NALP[CI_i], NVAL[CI_i], ms)
          for k in range(len(CGCs)):
            ci[CI_i][-1].coeffs.append(state_coeffs[CI_i][i][j]*CGCs[k])
            ci[CI_i][-1].occ.append(OCCs[k])
        ci[CI_i][-1].coeffs = numpy.array(ci[CI_i][-1].coeffs)
        ci[CI_i][-1].occ = numpy.array(ci[CI_i][-1].occ)
    
  #--- Calculating norm of CI states
  display('\nIn total, %d states have been read.' % sum([len(i) for i in ci]))
  display('Norm of the states:')
  for i in range(len(ci)):
    for j in range(len(ci[i])):
      norm = sum([x*x for x in ci[i][j].coeffs])
      display('\tState %s (%s,%s):\tNorm = %0.8f (%d Coefficients)' %
              (ci[i][j].info['state'],ci[i][j].info['spin'],ci[i][j].info['ms'],
               norm,len(ci[i][j].coeffs)))
  display('')
  
  ci_new = []
  for i in ci:
    ci_new.extend(i)
  
  return ci_new


def gamess_MOD(file_name,threshold=0.0,**kwargs):
  '''Reads GAMESS-US GUGA output.

  **Parameters:**

    file_name: str, file descriptor
      Specifies the filename for the input file.
      fname can also be used with a file descriptor instad of a filename.
    threshold : float, optional
      Specifies a read threshold for the CI coefficients.

  **Returns:**

    ci : list of CIinfo class instances
      See :ref:`Central Variables` for details.
  '''
  display('\nReading data of GUGA calculation from GAMESS-US...')

  ci = [[]]
  general_information = {'fileinfo': file_name,
                         'read_threshold': threshold}


  fname = open(file_name, 'r')


  from io import TextIOWrapper
  if isinstance(fname, TextIOWrapper):
    flines = fname.readlines()

  STATE_ZONE=False
  state_energies = []
  state_CSFs = []
  state_coeffs = []
  state_OCCs = []
  state_CVs = []

  NUMCI = 0
  NFZC = []
  NMCC = []
  NDOC = []
  NAOS = []
  NBOS = []
  NALP = []
  NVAL = []
  NEXT = []
  NFZV = []
  NACT = []
  NCORE = []
  nel = []
  spin = []
  cv_order = []

  for i,line in enumerate(flines):
    if "THE WAVEFUNCTION CONTAINS" in line:
      cv_order_i = flines[i+1].split() ### CHANGE TO ONE
      cv_order.append( [int(j) for j in cv_order_i[:NACT[-1]] if j != "0"]) 

    if " STATE #    1" in line:
      STATE_ZONE=True

    if STATE_ZONE:
      thisline = line.split()
      if " STATE #" in line:
        state_energies[-1].append(float(thisline[-1]))
        temp_state_CSFs = []
        temp_state_OCCs = []
        temp_state_CVs = []
        temp_state_coeffs = []
        continue
      if len(thisline) == 4 and abs(float(thisline[1])) >= threshold:
        temp_state_CSFs.append(int(thisline[0]))
        temp_state_coeffs.append(float(thisline[1]))
        temp = [int(x) for x in thisline[2]]
        temp_state_OCCs.append(temp)
        temp = [int(x) for x in thisline[3]]
        temp_state_CVs.append(temp)
        continue
      if " THE " in line:
        state_CSFs[-1].append(temp_state_CSFs)
        state_coeffs[-1].append(temp_state_coeffs)
        state_OCCs[-1].append(temp_state_OCCs)
        state_CVs[-1].append(temp_state_CVs)
        continue
      if "STEP CPU TIME" in line:
        STATE_ZONE = False
        continue

    if "CI CALCULATION" in line:
      NUMCI += 1
      ci.append([])
      state_energies.append([])
      state_CSFs.append([])
      state_coeffs.append([])
      state_OCCs.append([])
      state_CVs.append([])
      NFZC.append(int(flines[i+14].split()[1]))
      NDOC.append(int(flines[i+14].split()[3]))
      NEXT.append(int(flines[i+14].split()[5]))
      NMCC.append(int(flines[i+15].split()[1]))
      NAOS.append(int(flines[i+15].split()[3]))
      NFZV.append(int(flines[i+15].split()[5]))
      NBOS.append(int(flines[i+16].split()[1]))
      NALP.append(int(flines[i+17].split()[1]))
      NVAL.append(int(flines[i+18].split()[1]))
      NACT.append(NDOC[-1]+NAOS[-1]+NBOS[-1]+NALP[-1]+NVAL[-1]+NEXT[-1])
      NCORE.append(NMCC[-1]+NFZC[-1])
      nel.append( 2*(NFZC[-1]+NDOC[-1]+NMCC[-1]) + 1*(NAOS[-1]+NBOS[-1]+NALP[-1]))
      spin.append( 1 + NALP[-1] )
      continue

  fname.close()

  NCORE = [sum(i) for i in zip(*[NFZC,NMCC])]
  NACT = [sum(i) for i in zip(*[NDOC,NAOS,NBOS,NALP,NVAL,NEXT])]

  spinz = []
  for s in spin:
    spinz.append([s-1-2*x for x in range(s)])

  CSFs = []

  for i,drt in enumerate(state_CSFs):
    C = {}
    for j,s in enumerate(drt):
      for k,c in enumerate(s):  
        C[state_CSFs[i][j][k]] = state_CVs[i][j][k]
    CSFs.append(C)


  state_num = 0
  for CI_i in range(NUMCI):
    print(cv_order[CI_i]) #ZS
    for i in range(len(state_energies[CI_i])):
      for ms in spinz[CI_i]:
        ci[CI_i].append(CIinfo(method="guga"))
        ci[CI_i][-1].info = copy(general_information)
        ci[CI_i][-1].info["state"] = state_num
        state_num += 1
        ci[CI_i][-1].info["energy"] = state_energies[CI_i][i]
        ci[CI_i][-1].info["spin"] = spin[CI_i]
        ci[CI_i][-1].info["ms"] = ms/2
        ci[CI_i][-1].info["nel"] = nel[CI_i]
        ci[CI_i][-1].info["occ_info"] = [NCORE[CI_i],NACT[CI_i],NFZV[CI_i]]
        for j,csfi in enumerate(state_CSFs[CI_i][i]):
          CGCs, OCCs = CV_to_dets_MOD(CSFs[CI_i][csfi], ms, cv_order[CI_i], NCORE[CI_i], NCORE[CI_i]+NACT[CI_i])
          for k in range(len(CGCs)):
            ci[CI_i][-1].coeffs.append(state_coeffs[CI_i][i][j]*CGCs[k])
            ci[CI_i][-1].occ.append(OCCs[k])
        ci[CI_i][-1].coeffs = numpy.array(ci[CI_i][-1].coeffs)
        ci[CI_i][-1].occ = numpy.array(ci[CI_i][-1].occ)

    #--- Calculating norm of CI states
  display('\nIn total, %d states have been read.' % sum([len(i) for i in ci]))
  display('Norm of the states:')
  for i in range(len(ci)):
    for j in range(len(ci[i])):
      norm = sum([x*x for x in ci[i][j].coeffs])
      display('\tState %s (%s,%s):\tNorm = %0.8f (%d Coefficients)' %
              (ci[i][j].info['state'],ci[i][j].info['spin'],ci[i][j].info['ms'],
               norm,len(ci[i][j].coeffs)))
  display('')

  ci_new = []
  for i in ci:
    ci_new.extend(i)

  return ci_new

 
def CV_to_dets(CV, NCORE, NDOC, NALP, NVAL, MS):
    """
    Turn Case Vector into Determinants and Coefficients
    Not coded for NAOS or NBOS, can be adjusted
    GAMESS Case Vector ascends through unoccupied, descends through doubly occupied, the descends through singly occupied
    1 unoccupied
    2 single, raises total spin
    3 single, lowers total spin
    4 is doubly occupied ( necessarily singlet, see Szabo Ex. 2.38 )
    """
    
    NPHASE = 1
    IB = 0
    NOPEN = 0
    s = [] # Says if electron raises or lowers total spin (left to right)
    IORDER = [x for x in range(NCORE+NDOC+NALP+1, NCORE+NDOC+NALP+NVAL+1)] \
           + [x for x in range(NCORE+NDOC, NCORE, -1)] \
           + [x for x in range(NCORE+NDOC+NALP, NCORE+NDOC, -1)]
    
    for c in CV:
        if c == 1:
            continue
        if c == 2:
            IB += 1 
            NOPEN += 1
            s.append(1)
            continue

        if c == 3:
            IB -= 1
            NOPEN += 1
            s.append(-1)
            continue 

        if c == 4:
            if IB%2: 
                NPHASE *= -1 # Enforce Antisymmetry under Exchange?
            continue
            
    s_z, CGCs, iCGCs = SPNFNC(NOPEN, s, MS) # Have to go through all Ms when relativistic
    
    CGCs = [NPHASE*x for x in CGCs]
    iCGCs = [NPHASE*x for x in iCGCs]
    OCCs = []
    
    for i,sz in enumerate(s_z):
        DET = MAKDET(CV, IORDER, sz)
        occupation = numpy.zeros((NDOC+NALP+NVAL,2),dtype=numpy.intc)
        for orb in DET:
            if orb > 0:
                occupation[orb-NCORE-1,0] = 1
            elif orb < 0:
                occupation[abs(orb)-NCORE-1,1] = 1
        OCCs.append(occupation)
    
    return CGCs, OCCs

def CV_to_dets_MOD(CV, MS, IORDER, NCORE, NOCC):
    """
    Turn Case Vector into Determinants and Coefficients
    CV is the gamess unique case vector for CSF identification
    MS is the total spin projection, multiplied by 2
    IORDER is a list that says what MOs the elements of the case vector refer to
    NCORE is the number of core orbitals (always occupied)
    NOCC is the number of possibly occupied orbitals

    Not coded for NAOS or NBOS, can be adjusted
    GAMESS Case Vector ascends through unoccupied, descends through doubly occupied, the descends through singly occup
    1 unoccupied
    2 single, raises total spin
    3 single, lowers total spin
    4 is doubly occupied ( necessarily singlet, see Szabo Ex. 2.38 )
    """

    NPHASE = 1
    IB = 0
    NOPEN = 0
    s = [] # Says if electron raises or lowers total spin (left to right)
    
    for c in CV:
        if c == 1:
            continue
        if c == 2:
            IB += 1
            NOPEN += 1
            s.append(1)
            continue

        if c == 3:
            IB -= 1
            NOPEN += 1
            s.append(-1)
            continue

        if c == 4:
            if IB%2:
                NPHASE *= -1 # Enforce Antisymmetry under Exchange?
            continue
    s_z, CGCs, iCGCs = SPNFNC(NOPEN, s, MS) # Have to go through all Ms when relativistic

    CGCs = [NPHASE*x for x in CGCs]
    iCGCs = [NPHASE*x for x in iCGCs]
    OCCs = []

    for i,sz in enumerate(s_z):
        DET = MAKDET(CV, IORDER, sz)
        occupation = numpy.zeros((NOCC-NCORE,2),dtype=numpy.intc)
        for orb in DET:
            if orb > 0:
                occupation[orb-NCORE-1,0] = 1
            elif orb < 0:
                occupation[abs(orb)-NCORE-1,1] = 1
        OCCs.append(occupation)

    return CGCs, OCCs
        
        

def SPNFNC(NOPEN, s=[0], Ms=0):
    """
    Generates spin functions
    """
    
    if NOPEN == 0:
        return [0], [1], [1]

    s_z = [[1],[-1]]
    for i in range(NOPEN-1):
        s_z = [x+[1] for x in s_z] + [x+[-1] for x in s_z] # All permutations of spin_z for opens (off by factor of 2)
        
    S_z = []
    for sz in s_z:
        S_z.append([sum(sz[:1+i]) for i in range(NOPEN)]) # All total S_z as one travels along the spins
        
    pop_ind = 0
    while pop_ind < len(s_z):
        if S_z[pop_ind][-1] == Ms: # Exclude determinants with non-allowed total spin
            pop_ind += 1
        else:
            S_z.pop(pop_ind)
            s_z.pop(pop_ind)
            
    S = [sum(s[:1+i]) for i in range(NOPEN)] # Total Spin tracked from left to right
    
    pop_ind = 0
    while pop_ind < len(s_z):
        for j in range(NOPEN):
            if abs(S_z[pop_ind][j]) > S[j]: # Exclude determinants with non-allowed total spin couplings
                S_z.pop(pop_ind)
                s_z.pop(pop_ind)
                pop_ind -= 1
                break
        pop_ind += 1
    
    #   "Calculate the Clebsh-Gordon Coefficients"  
    
    CGCs = []
    iCGCs = []
    
    for i in range(len(s_z)):
        COEF = 1
        for j in range(NOPEN): 
            if s[j] == 1:
                if s_z[i][j] == 1:
                    COEF *= (S[j]+S_z[i][j])
                else:
                    COEF *= (S[j]-S_z[i][j])
                COEF /= 2*S[j]
            else:
                if s_z[i][j] == 1:
                    COEF *= (S[j]-S_z[i][j]+2)
                    if S[j]%2 == 0:
                        COEF *= -1
                else:
                    COEF *= (S[j]+S_z[i][j]+2)
                    if S[j]%2 == 1:
                        COEF *= -1
                COEF /= (2*S[j] + 4)
        if COEF == 0:
            continue
        CGCs.append(COEF/abs(COEF)**(1/2))
        iCGCs.append(1/COEF)
        
    return s_z, CGCs, iCGCs
    
def MAKDET(CV,IORDER,sz):
    DET = []
    iSO = 0
    for i,cv in enumerate(CV):
        if cv == 1:
            continue
        elif cv == 4:
            DET.append(IORDER[i])
            DET.append(-IORDER[i])
        else:
            if sz[iSO] == 1:
                DET.append(IORDER[i])
            else:
                DET.append(-IORDER[i])
            iSO += 1
    return DET
