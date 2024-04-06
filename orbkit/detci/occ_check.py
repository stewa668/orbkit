from . import cy_occ_check
from .. import omp_functions
from ..display import display
import numpy

def slice_occ(ij):
  '''Compares a slice of occupation patterns.
  '''
  if any([multici['cia'].method == i for i in ['mcscf','detci','fci','ci','guga']]):
    ab_sorting = multici['cia'].method == 'mcscf' # Alternating alpha beta orbitals
    ab_sorting = (ab_sorting or multici['cia'].method == 'guga') # Alternating alpha beta orbitals
    zero,sing = cy_occ_check.mcscf_ab(ij[0],ij[1],
                                      multici['cia'].coeffs,multici['cia'].occ,
                                      multici['cib'].coeffs,multici['cib'].occ,
                                      multici['moocc'],
                                      ab_sorting)
  elif any([multici['cia'].method == i for i in ['cis','tddft']]):
    zero,sing = cy_occ_check.cis_ab(ij[0],ij[1],
                                    multici['cia'].coeffs,multici['cia'].occ,
                                    multici['cib'].coeffs,multici['cib'].occ,
                                    multici['moocc'])
  return zero,sing

# ZS
# Make a spin-projected version
# May be issues from ab sorting...
def slice_occ_sp(ij):
  '''Compares a slice of occupation patterns.
  '''
  if any([multici['cia'].method == i for i in ['mcscf','detci','fci','ci','guga']]):
    ab_sorting = multici['cia'].method == 'mcscf' # Alternating alpha beta orbitals
    ab_sorting = (ab_sorting or multici['cia'].method == 'guga') # Alternating alpha beta orbitals
    zero,sing,zero_a,sing_a,zero_b,sing_b = cy_occ_check.mcscf_ab_sp(ij[0],ij[1],
                                              multici['cia'].coeffs,multici['cia'].occ,
                                              multici['cib'].coeffs,multici['cib'].occ,
                                              multici['moocc'],
                                              ab_sorting)
  else:
    print("Spin projected cis_ab not allowed. Ending run.")
    exit()
  return zero,sing,zero_a,sing_a,zero_b,sing_b

# ZS
# Make a spin flag.
def compare(cia,cib,spin_project=False,moocc=None,numproc=1):
  '''Compares occupation patterns of two CI vectors, and extracts identical 
  determinants and formal single excitations. 
  
  This is prerequisite for all subsequent detCI@orbkit calculations.
  
  **Parameters:**
  
    cia : CIinfo class instance 
      See :ref:`Central Variables` for details.
    cib : CIinfo class instance 
      See :ref:`Central Variables` for details.
    spin_project : Boolean
      Specifies if one wants info relevant for spin projected operators. (Spin-up electron density, ie.)
    moocc : None or numpy.array, dtype=numpy.intc, optional
      Specifies the closed molecular orbitals. If None, it has to be a member
      of `cia` AND `cib`.
    numproc : int
      Specifies number of subprocesses for multiprocessing.
      
  **Returns:**
  
    zero : list of two lists
      Identical Slater-determinants. 
      | First member: Prefactor (occupation * CI coefficient)
      | Second member: Indices of occupied orbitals
    sing : list of two lists
      Effective single excitation.
      |  First member: Product of CI coefficients
      |  Second member: Indices of the two molecular orbitals
    zero_a
    sing_a
    zero_b
    sing_b 
  '''
  global multici
  if moocc is None:
    moocc = cia.moocc
    assert moocc is not None, '`moocc` is not given'
    assert numpy.array_equal(moocc,cib.moocc), '`cia.moocc` has to be the same as `cib.moocc`'
  
  multici = {'cia': cia, 'cib': cib, 'moocc': moocc}
  
  numproc = min(len(cia.coeffs),max(1,numproc))
  ij = numpy.array(numpy.linspace(0, len(cia.coeffs), num=numproc+1, endpoint=True),  
                   dtype=numpy.intc) # backward compatibility
  ij = list(zip(ij[:-1],ij[1:]))
  
  display('\nComparing the occupation patterns \nof the determinants of the two states...')
  return_value = omp_functions.run(slice_occ,x=ij,numproc=numproc,display=display)

  if spin_project:
    return_value = omp_functions.run(slice_occ_sp,x=ij,numproc=numproc,display=display)
    zero = [[],[]]
    sing = [[],[]]
    zero_a = [[],[]]
    sing_a = [[],[]]
    zero_b = [[],[]]
    sing_b = [[],[]]
    for z,s,za,sa,zb,sb in return_value:
      zero[0].extend(z[0])
      zero[1].extend(z[1])
      sing[0].extend(s[0])
      sing[1].extend(s[1])
      zero_a[0].extend(za[0])
      zero_a[1].extend(za[1])
      sing_a[0].extend(sa[0])
      sing_a[1].extend(sa[1])
      zero_b[0].extend(zb[0])
      zero_b[1].extend(zb[1])
      sing_b[0].extend(sb[0])
      sing_b[1].extend(sb[1])
    return zero,sing,zero_a,sing_a,zero_b,sing_b
  else:
    return_value = omp_functions.run(slice_occ,x=ij,numproc=numproc,display=display)
    zero = [[],[]] 
    sing = [[],[]]
    for z,s in return_value:
      zero[0].extend(z[0])
      zero[1].extend(z[1])
      sing[0].extend(s[0])
      sing[1].extend(s[1])
    return zero,sing
