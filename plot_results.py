import matplotlib.pyplot as plt


def plot_n_unfairness(n_values, epsilon_values):
    plt.figure(figsize=(8, 6))

    cdf, cut, rank_100, rank_1000 = epsilon_values
    plt.plot(n_values, cdf, label="CDF", marker='^', markerfacecolor='none', markeredgecolor='orange', color='orange')
    plt.plot(n_values, cut, label="cut", marker='+', color='red')
    plt.plot(n_values, rank_100, label="rank-100", marker='s', markerfacecolor='none', markeredgecolor='green', color='green')
    plt.plot(n_values, rank_1000, label="rank-1000", marker='*', color='blue')

    plt.xlabel("Dataset size (n)")
    plt.ylabel("Unfairness (epsilon)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def plot_ratio_unfairness(ratios, epsilon_values):
    plt.figure(figsize=(8, 6))

    cdf, cut, rank_100, rank_1000 = epsilon_values
    plt.plot(ratios, cdf, label="CDF", marker='^', markerfacecolor='none', markeredgecolor='orange', color='orange')
    plt.plot(ratios, cut, label="cut", marker='+', color='red')
    plt.plot(ratios, rank_100, label="rank-100", marker='s', markerfacecolor='none', markeredgecolor='green', color='green')
    plt.plot(ratios, rank_1000, label="rank-1000", marker='*', color='blue')

    plt.xlabel("Ratio")
    plt.ylabel("Unfairness (epsilon)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def plot_m_unfairness(m_values, epsilon_values):
    plt.figure(figsize=(8, 6))

    cdf, cut, rank_100, rank_1000 = epsilon_values
    plt.plot(m_values, cdf, label="CDF", marker='^', markerfacecolor='none', markeredgecolor='orange', color='orange')
    plt.plot(m_values, cut, label="cut", marker='+', color='red')
    plt.plot(m_values, rank_100, label="rank-100", marker='s', markerfacecolor='none', markeredgecolor='green', color='green')
    plt.plot(m_values, rank_1000, label="rank-1000", marker='*', color='blue')

    plt.xlabel("Number of buckets (m)")
    plt.ylabel("Unfairness (epsilon)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def plot_n_space(n_values, number_of_cuts):
    plt.figure(figsize=(8, 6))

    cdf, cut, rank_100, rank_1000 = number_of_cuts
    plt.plot(n_values, cdf, label="CDF", marker='^', markerfacecolor='none', markeredgecolor='orange', color='orange')
    plt.plot(n_values, cut, label="cut", marker='+', color='red')
    plt.plot(n_values, rank_100, label="rank-100", marker='s', markerfacecolor='none', markeredgecolor='green', color='green')
    plt.plot(n_values, rank_1000, label="rank-1000", marker='*', color='blue')

    plt.xlabel("Dataset size (n)")
    plt.ylabel("Number of cuts")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def plot_ratio_space(n_values, number_of_cuts):
    plt.figure(figsize=(8, 6))

    cdf, cut, rank_100, rank_1000 = number_of_cuts
    plt.plot(n_values, cdf, label="CDF", marker='^', markerfacecolor='none', markeredgecolor='orange', color='orange')
    plt.plot(n_values, cut, label="cut", marker='+', color='red')
    plt.plot(n_values, rank_100, label="rank-100", marker='s', markerfacecolor='none', markeredgecolor='green', color='green')
    plt.plot(n_values, rank_1000, label="rank-1000", marker='*', color='blue')

    plt.xlabel("Ratio")
    plt.ylabel("Number of cuts")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def plot_m_space(n_values, number_of_cuts):
    plt.figure(figsize=(8, 6))

    cdf, cut, rank_100, rank_1000 = number_of_cuts
    plt.plot(n_values, cdf, label="CDF", marker='^', markerfacecolor='none', markeredgecolor='orange', color='orange')
    plt.plot(n_values, cut, label="cut", marker='+', color='red')
    plt.plot(n_values, rank_100, label="rank-100", marker='s', markerfacecolor='none', markeredgecolor='green', color='green')
    plt.plot(n_values, rank_1000, label="rank-1000", marker='*', color='blue')

    plt.xlabel("Number of buckets (m)")
    plt.ylabel("Number of cuts")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def plot_m_query(n_values, number_of_cuts):
    plt.figure(figsize=(8, 6))

    cdf, cut, rank_100, rank_1000 = number_of_cuts
    plt.plot(n_values, cdf, label="CDF", marker='^', markerfacecolor='none', markeredgecolor='orange', color='orange')
    plt.plot(n_values, cut, label="cut", marker='+', color='red')
    plt.plot(n_values, rank_100, label="rank-100", marker='s', markerfacecolor='none', markeredgecolor='green', color='green')
    plt.plot(n_values, rank_1000, label="rank-1000", marker='*', color='blue')

    plt.xlabel("Number of buckets (m)")
    plt.ylabel("Time (sec)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


