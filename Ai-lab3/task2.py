import matplotlib.pyplot as plt

def trimf(x, params):
    a, b, c = params
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    elif b < x < c:
        return (c - x) / (c - b) if c != b else 1.0
    return 0.0

def simulate_ac(temp_val, rate_val):
    t_vc = trimf(temp_val, [0, 0, 15])
    t_c = trimf(temp_val, [10, 18, 22])
    t_n = trimf(temp_val, [20, 24, 28])
    t_w = trimf(temp_val, [26, 30, 35])
    t_vw = trimf(temp_val, [32, 40, 40])

    r_neg = trimf(rate_val, [-5, -5, 0])
    r_z = trimf(rate_val, [-2, 0, 2])
    r_pos = trimf(rate_val, [0, 5, 5])

    angle_act = {
        'large_left': 0, 'small_left': 0, 'zero': 0,
        'small_right': 0, 'large_right': 0
    }

    w1 = min(t_vw, r_pos)
    angle_act['large_left'] = max(angle_act['large_left'], w1)
    
    w2 = min(t_vw, r_neg)
    angle_act['small_left'] = max(angle_act['small_left'], w2)
    
    w3 = min(t_w, r_pos)
    angle_act['large_left'] = max(angle_act['large_left'], w3)
    
    w4 = min(t_w, r_neg)
    angle_act['zero'] = max(angle_act['zero'], w4)
    
    w5 = min(t_vc, r_neg)
    angle_act['large_right'] = max(angle_act['large_right'], w5)
    
    w6 = min(t_vc, r_pos)
    angle_act['small_right'] = max(angle_act['small_right'], w6)
    
    w7 = min(t_c, r_neg)
    angle_act['large_right'] = max(angle_act['large_right'], w7)
    
    w8 = min(t_c, r_pos)
    angle_act['zero'] = max(angle_act['zero'], w8)
    
    w9 = min(t_vw, r_z)
    angle_act['large_left'] = max(angle_act['large_left'], w9)
    
    w10 = min(t_w, r_z)
    angle_act['small_left'] = max(angle_act['small_left'], w10)
    
    w11 = min(t_vc, r_z)
    angle_act['large_right'] = max(angle_act['large_right'], w11)
    
    w12 = min(t_c, r_z)
    angle_act['small_right'] = max(angle_act['small_right'], w12)
    
    w13 = min(t_n, r_pos)
    angle_act['small_left'] = max(angle_act['small_left'], w13)
    
    w14 = min(t_n, r_neg)
    angle_act['small_right'] = max(angle_act['small_right'], w14)
    
    w15 = min(t_n, r_z)
    angle_act['zero'] = max(angle_act['zero'], w15)

    terms_params = {
        'large_left': [-90, -90, -45],
        'small_left': [-60, -30, 0],
        'zero': [-10, 0, 10],
        'small_right': [0, 30, 60],
        'large_right': [45, 90, 90]
    }

    num = 0.0
    den = 0.0
    for x in range(-90, 91):
        max_mu = 0.0
        for term, act_val in angle_act.items():
            if act_val > 0:
                mu = min(act_val, trimf(x, terms_params[term]))
                max_mu = max(max_mu, mu)
        num += x * max_mu
        den += max_mu
    
    result_angle = num / den if den != 0 else 0.0
    
    return result_angle, angle_act, terms_params

def plot_fuzzy_result(act_dict, terms_params, result_angle, title):
    x_vals = list(range(-90, 91))
    plt.figure(figsize=(8, 4))
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
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
    plt.xlabel("Кут повороту регулятора кондиціонера")
    plt.ylabel("Ступінь належності")
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.show()

test_temp = 35 
test_rate = 2  

res_angle, act_dict, t_params = simulate_ac(test_temp, test_rate)

print("Задача 2: Керування кондиціонером")
print(f"Вхід: Температура = {test_temp}°C, Швидкість зміни = {test_rate}")
print(f"Поворот регулятора: {res_angle:.2f} градусів")

plot_fuzzy_result(act_dict, t_params, res_angle, "Регулятор кондиціонера")