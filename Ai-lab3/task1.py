import matplotlib.pyplot as plt

def xor(x1, x2):
    return (x1 or x2) and not (x1 and x2)

def trimf(x, params):
    a, b, c = params
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    elif b < x < c:
        return (c - x) / (c - b) if c != b else 1.0
    return 0.0

def simulate_mixer(temp_val, flow_val):
    t_cold = trimf(temp_val, [0, 0, 25])
    t_cool = trimf(temp_val, [15, 30, 45])
    t_warm = trimf(temp_val, [35, 50, 65])
    t_not_very_hot = trimf(temp_val, [55, 75, 90])
    t_hot = trimf(temp_val, [80, 100, 100])

    p_weak = trimf(flow_val, [0, 0, 40])
    p_not_very_strong = trimf(flow_val, [30, 50, 70])
    p_strong = trimf(flow_val, [60, 100, 100])

    hot_act = {
        'large_left': 0, 'medium_left': 0, 'small_left': 0, 'zero': 0,
        'small_right': 0, 'medium_right': 0, 'large_right': 0
    }
    cold_act = {
        'large_left': 0, 'medium_left': 0, 'small_left': 0, 'zero': 0,
        'small_right': 0, 'medium_right': 0, 'large_right': 0
    }

    w1 = min(t_hot, p_strong)
    hot_act['medium_left'] = max(hot_act['medium_left'], w1)
    cold_act['medium_right'] = max(cold_act['medium_right'], w1)
    
    w2 = min(t_hot, p_not_very_strong)
    hot_act['zero'] = max(hot_act['zero'], w2)
    cold_act['medium_right'] = max(cold_act['medium_right'], w2)
    
    w3 = min(t_not_very_hot, p_strong)
    hot_act['small_left'] = max(hot_act['small_left'], w3)
    cold_act['zero'] = max(cold_act['zero'], w3)
    
    w4 = min(t_not_very_hot, p_weak)
    hot_act['small_right'] = max(hot_act['small_right'], w4)
    cold_act['small_right'] = max(cold_act['small_right'], w4)
    
    w5 = min(t_warm, p_not_very_strong)
    hot_act['zero'] = max(hot_act['zero'], w5)
    cold_act['zero'] = max(cold_act['zero'], w5)
    
    w6 = min(t_cool, p_strong)
    hot_act['medium_right'] = max(hot_act['medium_right'], w6)
    cold_act['medium_left'] = max(cold_act['medium_left'], w6)
    
    w7 = min(t_cool, p_not_very_strong)
    hot_act['medium_right'] = max(hot_act['medium_right'], w7)
    cold_act['small_left'] = max(cold_act['small_left'], w7)
    
    w8 = min(t_cold, p_weak)
    hot_act['large_right'] = max(hot_act['large_right'], w8)
    cold_act['zero'] = max(cold_act['zero'], w8)
    
    w9 = min(t_cold, p_strong)
    hot_act['medium_left'] = max(hot_act['medium_left'], w9)


    cold_act['medium_right'] = max(cold_act['medium_right'], w9)
    
    w10 = min(t_warm, p_strong)
    hot_act['small_left'] = max(hot_act['small_left'], w10)
    cold_act['small_left'] = max(cold_act['small_left'], w10)
    
    w11 = min(t_warm, p_weak)
    hot_act['small_right'] = max(hot_act['small_right'], w11)
    cold_act['small_right'] = max(cold_act['small_right'], w11)

    terms_params = {
        'large_left': [-90, -90, -60], 'medium_left': [-75, -45, -15],
        'small_left': [-30, -15, 0],   'zero': [-10, 0, 10],
        'small_right': [0, 15, 30],    'medium_right': [15, 45, 75],
        'large_right': [60, 90, 90]
    }

    def defuzzify(act_dict):
        num = 0.0
        den = 0.0
        for x in range(-90, 91):
            max_mu = 0.0
            for term, act_val in act_dict.items():
                if act_val > 0:
                    mu = min(act_val, trimf(x, terms_params[term]))
                    max_mu = max(max_mu, mu)
            num += x * max_mu
            den += max_mu
        return num / den if den != 0 else 0.0

    hot_angle = defuzzify(hot_act)
    cold_angle = defuzzify(cold_act)
    
    return hot_angle, cold_angle, hot_act, terms_params

def plot_fuzzy_result(act_dict, terms_params, result_angle, title):
    x_vals = list(range(-90, 91))
    plt.figure(figsize=(8, 4))
    
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
    color_idx = 0
    
    for term, params in terms_params.items():
        y_vals = [trimf(x, params) for x in x_vals]


        c = colors[color_idx % len(colors)]
      
        plt.plot(x_vals, y_vals, label=term, color=c, linestyle='-', linewidth=1.5, alpha=0.6)
        
        act_val = act_dict.get(term, 0)
        if act_val > 0:
            y_act = [min(act_val, y) for y in y_vals]
            plt.fill_between(x_vals, 0, y_act, color=c, alpha=0.4)
            
        color_idx += 1
            
    plt.axvline(x=result_angle, color='black', linewidth=3, label=f'Результат: {result_angle:.2f}°')
    
    plt.title(title)
    plt.xlabel("Кут повороту")
    plt.ylabel("Ступінь належності")
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.show()


print(f"0 XOR 1 = {int(xor(False, True))}")
print(f"1 XOR 1 = {int(xor(True, True))}\n")

hot_result, cold_result, hot_act, terms_params = simulate_mixer(10, 20)
print("Задача 1: Керування кранами")
print(f"Поворот гарячого крану: {hot_result:.2f} градусів")
print(f"Поворот холодного крану: {cold_result:.2f} градусів")

plot_fuzzy_result(hot_act, terms_params, hot_result, "Кран гарячої води")
