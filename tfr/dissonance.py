import matplotlib.pyplot as plt
import seaborn as sns

%pylab


# wave A
plot(t, sine(t, 10))
# wave B
plot(t, sine(t, 11))
# wave A + B
plot(t, sine(t, 10) + sine(t, 11))
# envelope waves
plot(t, sine(t, 0.5, 2, 0.5 * np.pi))
plot(t, sine(t, 0.5, 2, 1.5 * np.pi))
