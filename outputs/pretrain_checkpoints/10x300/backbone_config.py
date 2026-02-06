backbone_config = {
        'd_byte': 128,
        'd_packet': 260,
        'byte_layers': 3,
        'packet_layers': 4,
        'num_heads': 5,
        'd_ff': 1024,
        'max_bytes': 300,
        'max_packets': 100,
        'dropout': 0.1,
        'use_stats': False,
        'stats_dim': 36,
        'use_adaptive_gating': True
    }

data_max_packets = 10
data_max_bytes = 300
output_dir_suffix = "10x300"