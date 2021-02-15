import numpy as np
import matplotlib.pyplot as plt


def add_noise(pattern, noise_frac=0.0, seed=42):
    flips_num = int(pattern.shape[0] * noise_frac)
    to_flip = np.random.default_rng(seed).choice(
        pattern.shape[0], size=(flips_num), replace=False)
    noisy_pattern = pattern.copy()
    noisy_pattern[to_flip] = noisy_pattern[to_flip]*-1
    return noisy_pattern


def plot_distortion_resistance(hop_net, patterns, pattern_num, steps=1):
    pattern = patterns[pattern_num]
    is_noise_removed = []
    noise_levels = np.linspace(0, 1, 101)
    for noise in noise_levels:
        is_removed_seeds = []
        for seed in range(50):
            noisy_pattern = add_noise(pattern, noise, seed=seed)
            inter, patt, energy = hop_net.update_rule(
                noisy_pattern, steps, verbose=False)
            is_removed_seeds.append(np.array_equal(pattern, patt))
                
        is_noise_removed.append(np.mean(is_removed_seeds))

    plt.plot(noise_levels, is_noise_removed)
    plt.xlabel('Noise level')
    plt.ylabel('Converged frac.')
    plt.title(
        f'Distortion resistance for pattern {pattern_num}, mean: {round(np.mean(is_noise_removed), 4)}')
        # f'Distortion resistance for pattern {pattern_num}, until {np.where(np.array(is_noise_removed) < 1)[0][0]}% noise')


def plot_attractor_for_pattern(hop_net, patterns, pattern_num, noise, steps=1):
    pattern = patterns[pattern_num]
    noisy_pattern = add_noise(pattern, noise)
    inter, patt, energy = hop_net.update_rule(
        noisy_pattern, steps, verbose=False)
    plt.imshow(patt.reshape((32, 32)), cmap="binary")
    return patt
