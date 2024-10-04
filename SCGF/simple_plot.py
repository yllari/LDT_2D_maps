import numpy as np
import matplotlib.pyplot as plt
path = "SCGF.dat"
data = np.genfromtxt(path)
alpha = data[:,0]
theta = data[:,1]

diff = np.diff(theta)
print(theta)

d_theta = [(theta[i+1]-theta[i-1])/((alpha[i+1]-alpha[i-1])) for i in
        range(1,len(theta)-1)]
dd_theta = [(theta[i+1]-2*theta[i]+theta[i-1])/((alpha[i+1]-alpha[i])**2) for i in
        range(1,len(theta)-1)]

fig, ax = plt.subplots(1,2)

ax[0].plot(alpha, theta, "o")
ax[1].plot(alpha[1:-1],d_theta,"o-")

#plt.xticks([0,0.25,0.5,0.75,1])
ax[0].set_xlabel(r"$\alpha$")
ax[0].set_ylabel(r"$\theta(\alpha)$")
ax[1].set_ylabel(r"$\theta'=<A>$")
ax[1].set_xlabel(r"$I'(a)=\alpha$")
fig.tight_layout()
fig.set_size_inches(10, 5)
fig.savefig("SCGF_first_deriv.png",dpi=300)
