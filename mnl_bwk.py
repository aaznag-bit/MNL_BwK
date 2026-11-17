from math import floor, log, log1p, sqrt
from random import Random
from time import perf_counter


def confidence_bounds(retained, sales, logarithm):
    if retained == 0:
        return 0.0, 1.0
    mean = sales / retained
    a = 2 * logarithm / retained
    b = a / 3
    w = mean - b
    lower = 0.0 if w <= 0 else w * w / (w + a / 2 + sqrt(a * a / 4 + a * w))
    upper = min(0.5, mean + b + a / 2 + sqrt(a * a / 4 + a * (mean + b)))
    if lower > upper:
        return 0.0, 1.0
    return lower / (1 - lower), upper / (1 - upper)


def fluid_lp(utilities, capacities, cardinality=4):
    n = len(utilities)
    order = sorted(range(n), key=lambda i: (-utilities[i], i))
    top = [i for i in order[:cardinality] if utilities[i] > 0]
    z = 1 / (1 + sum(utilities[i] for i in top))
    if all(utilities[i] * z <= capacities[i] + 1e-14 for i in top):
        return 1 - z, [(sum(1 << i for i in top), 1.0)]
    for _ in range(100):
        remaining_a, remaining_b = cardinality, 0.0
        objective_a = objective_b = 0.0
        x = [0.0] * n
        for i in order:
            if utilities[i] <= 0 or capacities[i] <= 0:
                continue
            value, a, b = z, 1.0, 0.0
            cap = capacities[i] / utilities[i]
            if cap < value:
                value, a, b = cap, 0.0, cap
            remaining = remaining_a * z + remaining_b
            if remaining < value:
                value, a, b = remaining, remaining_a, remaining_b
            if value <= 1e-15:
                continue
            x[i] = value
            remaining_a -= a
            remaining_b -= b
            objective_a += utilities[i] * a
            objective_b += utilities[i] * b
        next_z = (1 - objective_b) / (1 + objective_a)
        if abs(next_z - z) < 2e-14:
            break
        z = next_z
    else:
        raise ArithmeticError("Fluid LP did not converge")
    cumulative = [0.0]
    knots = {0.0, 1.0}
    for value in x:
        probability = value / z
        if probability < 1e-13:
            probability = 0.0
        elif probability > 1 - 1e-13:
            probability = 1.0
        cumulative.append(cumulative[-1] + probability)
        fraction = cumulative[-1] % 1
        if 1e-13 < fraction < 1 - 1e-13:
            knots.add(fraction)
    knots = sorted(knots)
    law = {}
    for left, right in zip(knots, knots[1:]):
        if right - left <= 1e-13:
            continue
        midpoint = (left + right) / 2
        subset = [i for i in range(n)
                  if floor(cumulative[i + 1] - midpoint) > floor(cumulative[i] - midpoint)]
        mask = sum(1 << i for i in subset)
        mass = z * (right - left) * (1 + sum(utilities[i] for i in subset))
        law[mask] = law.get(mask, 0.0) + mass
    if abs(sum(law.values()) - 1) > 2e-10:
        raise ArithmeticError("Invalid fluid LP distribution")
    return sum(u * value for u, value in zip(utilities, x)), list(law.items())


def split_assortments(candidate_law, lower, upper):
    law = {}
    for candidate, probability in candidate_law:
        if candidate == 0:
            continue
        items = [i for i in range(len(upper)) if candidate & (1 << i)]
        denominator = 1 + sum(upper[i] for i in items)
        group, width, lower_sum = 0, 0.0, 0.0
        groups = []
        for i in items:
            group |= 1 << i
            width += upper[i] - lower[i]
            lower_sum += lower[i]
            if width >= 1:
                groups.append((group, lower_sum))
                group, width, lower_sum = 0, 0.0, 0.0
        if group:
            groups.append((group, lower_sum))
        for group, lower_sum in groups:
            mass = probability * (1 + lower_sum) / denominator
            law[group] = law.get(group, 0.0) + mass
    if sum(law.values()) > 1 + 2e-10:
        raise ArithmeticError("Invalid assortment sampling probabilities")
    return list(law.items())


def instance(regime, horizon, epsilon, parameter):
    if horizon < 2 or regime not in {"infinite", "sublinear", "scaling"}:
        raise ValueError("Invalid experiment configuration")
    n = 10
    utilities = [(0.25 if regime == "infinite" else 0.75)
                 + (epsilon if i in (0, 1, 8, 9) else 0) for i in range(n)]
    if not all(0 <= u <= 1 for u in utilities):
        raise ValueError("Utilities must lie in [0, 1]")
    if regime == "infinite":
        inventory = [horizon] * n
    else:
        scale = horizon ** (1 + parameter) if regime == "sublinear" else horizon * parameter
        inventory = [floor((i + 1) * scale / n + 1e-9) for i in range(n)]
    if not all(0 <= b <= horizon for b in inventory):
        raise ValueError("Inventory must lie between zero and the horizon")
    return utilities, inventory


def simulate(regime, algorithm, horizon, epsilon, parameter, seed):
    started = perf_counter()
    utilities, inventory = instance(regime, horizon, epsilon, parameter)
    n, cardinality = len(utilities), 4
    logarithm = log(8 * n) + 4 * log(horizon + 1)
    active = [b > 256 * logarithm for b in inventory]
    capacities = [(b - 8 * sqrt(b * logarithm) - 8 * logarithm - 2) / horizon
                  if included else 0.0 for b, included in zip(inventory, active)]
    lower = [0.0] * n
    upper = [float(included) for included in active]
    sales, retained = [0] * n, [0] * n
    benchmark = horizon * fluid_lp(utilities, [b / horizon for b in inventory], cardinality)[0]
    utility_sums = [0.0] * (1 << n)
    rewards = [0.0] * (1 << n)
    subsets = [()] * (1 << n)
    for mask in range(1, 1 << n):
        i = (mask & -mask).bit_length() - 1
        utility_sums[mask] = utility_sums[mask ^ (1 << i)] + utilities[i]
        rewards[mask] = utility_sums[mask] / (1 + utility_sums[mask])
        subsets[mask] = (i,) + subsets[mask ^ (1 << i)]
    rng = Random(seed)

    def choice(mask):
        draw = rng.random() * (1 + utility_sums[mask]) - 1
        if draw < 0:
            return -1
        for i in subsets[mask]:
            draw -= utilities[i]
            if draw < 0:
                return i
        return subsets[mask][-1]

    expected_revenue = 0.0
    time = stop_time = offers = lp_solves = 0
    if algorithm == "AAGZ":
        if regime != "infinite":
            raise ValueError("AAGZ is used only in the infinite-inventory experiment")
        upper = [1.0] * n
        epochs, epoch_totals = [0] * n, [0] * n
        epoch = 1
        while time < horizon:
            assortment = sorted(range(n), key=lambda i: (-upper[i], i))[:cardinality]
            mask = sum(1 << i for i in assortment)
            epoch_sales = [0] * n
            while time < horizon:
                time += 1
                offers += 1
                expected_revenue += rewards[mask]
                purchased = choice(mask)
                if purchased < 0:
                    radius_factor = 48 * log(sqrt(n) * epoch + 1)
                    for i in assortment:
                        epochs[i] += 1
                        epoch_totals[i] += epoch_sales[i]
                        mean = epoch_totals[i] / epochs[i]
                        radius = radius_factor / epochs[i]
                        upper[i] = mean + sqrt(mean * radius) + radius
                        retained[i] += 1
                    epoch += 1
                    break
                epoch_sales[purchased] += 1
                sales[purchased] += 1
                retained[purchased] += 1
    elif algorithm == "MNLwK-UCB":
        while time < horizon and any(active):
            _, candidate_law = fluid_lp(upper, capacities, cardinality)
            offered_law = split_assortments(candidate_law, lower, upper)
            lp_solves += 1
            mass = sum(probability for _, probability in offered_law)
            if mass <= 0:
                break
            skipped = floor(log1p(-rng.random()) / log1p(-mass)) if mass < 1 else 0
            if skipped >= horizon - time:
                break
            time += skipped + 1
            draw = rng.random() * mass
            mask = offered_law[-1][0]
            for candidate, probability in offered_law:
                draw -= probability
                if draw < 0:
                    mask = candidate
                    break
            offers += 1
            expected_revenue += rewards[mask]
            purchased = choice(mask)
            if purchased < 0:
                for i in subsets[mask]:
                    retained[i] += 1
                    lower[i], upper[i] = confidence_bounds(retained[i], sales[i], logarithm)
            else:
                retained[purchased] += 1
                sales[purchased] += 1
                if sales[purchased] == inventory[purchased]:
                    stop_time = time
                    break
                lower[purchased], upper[purchased] = confidence_bounds(
                    retained[purchased], sales[purchased], logarithm)
    else:
        raise ValueError("Unknown algorithm")
    if any(sold > stock for sold, stock in zip(sales, inventory)):
        raise ArithmeticError("Inventory constraint violated")
    result = dict(regime=regime, algorithm=algorithm, T=horizon, epsilon=epsilon,
                  parameter=parameter, seed=seed, benchmark=benchmark,
                  expected_revenue=expected_revenue, realized_revenue=sum(sales),
                  regret=benchmark - expected_revenue, realized_regret=benchmark - sum(sales),
                  active_products=n if algorithm == "AAGZ" else sum(active),
                  stop_time=stop_time, nonempty_offers=offers, lp_solves=lp_solves,
                  seconds=perf_counter() - started)
    for name, values in (("B", inventory), ("sales", sales), ("retained", retained)):
        result.update({f"{name}{i + 1}": value for i, value in enumerate(values)})
    return result
