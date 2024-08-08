def self_add(dict):
    dict['x'] += 1

config = {
    'x': 1
}
self_add(config)
print(config['x']) 