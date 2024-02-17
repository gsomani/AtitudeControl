def rotationMatrixSingleAxis(w,i):
 z = np.exp(1j*w)
 M = np.eye(3)
  
 j,k = [(i+a)%3 for a in range(2)]
 
 M[j,j] = M[k,k] = z.real 
 M[j,k] = -z.imag
 M[k,j] = z.imag
  
 return M 

def rotationMatrixFromOmega(omega, dt):
 M = np.eye(3)
 for i in range(3):
  A = rotationMatrixSingleAxis(omega*dt, i)
  M = np.dot(A,M)

 return M


