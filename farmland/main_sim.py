filename = "main"

sim_unlocks = Unlocks
sim_items={Items.Wood: 99999999, Items.Carrot : 1000000, Items.Hay : 5000000, Items.Fertilizer: 9999999, Items.Power: 999999999}
sim_globals={}
seed=0
speedup=1024
run_time = simulate(filename, sim_unlocks, sim_items, sim_globals, seed, speedup)

