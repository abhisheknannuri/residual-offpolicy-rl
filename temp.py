# import wandb
# api = wandb.Api()
# run = api.run("nannuriabhi2000-hochschule-schmalkalden/dexmg-bc/kpid6s4r")
# for art in run.logged_artifacts():
#     print(art.name, art.state)




# from collections import defaultdict
# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# ds = LeRobotDataset("poolvarine/robomimic-mh-lift-image-dense")


# # Print dataset keys immediately
# print("Dataset Features:", list(ds.features.keys()))


# episode_lengths = defaultdict(int)

# # Count steps per episode
# for row in ds:
#     # THE FIX: Extract the raw integer from the PyTorch Tensor
#     eid = row["episode_index"].item() 
#     episode_lengths[eid] += 1

# # Sort by length (descending)
# sorted_lengths = sorted(episode_lengths.items(), key=lambda x: x[1], reverse=True)

# print("Number of episodes:", len(episode_lengths))
# print("\nTop 10 longest episodes:")
# for eid, length in sorted_lengths[:10]:
#     print(f"Episode {eid}: {length} steps")



"""

"ankile/robomimic-mh-lift-image" dataset stats:

Number of episodes: 300
First 10 episode lengths: [(0, 48), (1, 55), (2, 46), (3, 45), (4, 43), (5, 50), (6, 46), (7, 52), (8, 51), (9, 41), (10, 49), (11, 48), (12, 48), (13, 49), (14, 58), (15, 55), (16, 46), (17, 54), (18, 46), (19, 48), (20, 59), (21, 45), (22, 44), (23, 52), (24, 48), (25, 56), (26, 47), (27, 51), (28, 51), (29, 50), (30, 44), (31, 51), (32, 52), (33, 49), (34, 57), (35, 52), (36, 53), (37, 48), (38, 42), (39, 55), (40, 49), (41, 46), (42, 44), (43, 44), (44, 50), (45, 53), (46, 55), (47, 45), (48, 48), (49, 48), (50, 213), (51, 179), (52, 180), (53, 170), (54, 155), (55, 156), (56, 136), (57, 136), (58, 171), (59, 192), (60, 149), (61, 185), (62, 146), (63, 154), (64, 185), (65, 165), (66, 147), (67, 152), (68, 161), (69, 197), (70, 171), (71, 170), (72, 168), (73, 172), (74, 154), (75, 174), (76, 140), (77, 152), (78, 157), (79, 269), (80, 164), (81, 161), (82, 138), (83, 210), (84, 138), (85, 163), (86, 150), (87, 142), (88, 156), (89, 140), (90, 169), (91, 154), (92, 146), (93, 145), (94, 195), (95, 161), (96, 165), (97, 173), (98, 185), (99, 219), (100, 118), (101, 56), (102, 78), (103, 71), (104, 82), (105, 55), (106, 72), (107, 72), (108, 109), (109, 81), (110, 72), (111, 68), (112, 88), (113, 69), (114, 84), (115, 106), (116, 94), (117, 84), (118, 91), (119, 101), (120, 73), (121, 71), (122, 82), (123, 86), (124, 112), (125, 62), (126, 76), (127, 77), (128, 90), (129, 79), (130, 67), (131, 74), (132, 66), (133, 72), (134, 97), (135, 74), (136, 64), (137, 72), (138, 77), (139, 82), (140, 90), (141, 64), (142, 75), (143, 77), (144, 97), (145, 115), (146, 68), (147, 81), (148, 78), (149, 74), (150, 101), (151, 108), (152, 95), (153, 93), (154, 96), (155, 97), (156, 96), (157, 103), (158, 102), (159, 108), (160, 92), (161, 91), (162, 100), (163, 90), (164, 95), (165, 105), (166, 104), (167, 80), (168, 90), (169, 95), (170, 95), (171, 91), (172, 88), (173, 94), (174, 95), (175, 95), (176, 97), (177, 90), (178, 81), (179, 101), (180, 88), (181, 89), (182, 93), (183, 99), (184, 89), (185, 100), (186, 100), (187, 97), (188, 100), (189, 88), (190, 97), (191, 94), (192, 88), (193, 95), (194, 86), (195, 89), (196, 108), (197, 91), (198, 115), (199, 88), (200, 125), (201, 76), (202, 166), (203, 100), (204, 140), (205, 106), (206, 118), (207, 126), (208, 91), (209, 82), (210, 89), (211, 96), (212, 124), (213, 182), (214, 119), (215, 82), (216, 96), (217, 88), (218, 87), (219, 84), (220, 122), (221, 106), (222, 149), (223, 93), (224, 110), (225, 114), (226, 107), (227, 106), (228, 210), (229, 125), (230, 150), (231, 133), (232, 131), (233, 122), (234, 111), (235, 156), (236, 234), (237, 106), (238, 95), (239, 107), (240, 107), (241, 303), (242, 121), (243, 133), (244, 118), (245, 120), (246, 140), (247, 158), (248, 159), (249, 80), (250, 100), (251, 72), (252, 118), (253, 96), (254, 140), (255, 129), (256, 101), (257, 96), (258, 132), (259, 313), (260, 100), (261, 87), (262, 101), (263, 102), (264, 97), (265, 94), (266, 137), (267, 80), (268, 106), (269, 98), (270, 77), (271, 105), (272, 104), (273, 95), (274, 86), (275, 81), (276, 99), (277, 112), (278, 183), (279, 101), (280, 118), (281, 78), (282, 95), (283, 132), (284, 110), (285, 91), (286, 89), (287, 96), (288, 112), (289, 72), (290, 99), (291, 97), (292, 93), (293, 92), (294, 66), (295, 90), (296, 114), (297, 134), (298, 96), (299, 127)]
Top 10 longest episodes: [(259, 313), (241, 303), (79, 269), (236, 234), (99, 219), (50, 213), (83, 210), (228, 210), (69, 197), (94, 195)]

"""


# import inspect
# from robosuite.environments.manipulation.pick_place import PickPlaceCan, PickPlace


# def print_can_reward_numeric_summary(reward_scale: float = 1.0, single_object_mode: int = 2) -> None:
# 		"""Print practical reward range summary for robosuite PickPlaceCan.

# 		single_object_mode:
# 			0 -> all objects
# 			1/2 -> single object task (Can task commonly uses this)
# 		"""
# 		# From PickPlace.staged_rewards()
# 		reach_max = 0.1
# 		grasp_max = 0.35
# 		lift_max = 0.5
# 		hover_max = 0.7

# 		# Dense shaping uses max(staged_rewards), so pre-success shaped max is hover_max.
# 		shaped_pre_success_raw_max = hover_max

# 		# Sparse success term from reward(): reward = sum(objects_in_bins)
# 		# For single-object mode this is at most 1. For mode=0 there are 4 objects.
# 		sparse_success_raw_max = 4.0 if single_object_mode == 0 else 1.0

# 		# In mode=0, reward() divides by 4 after scaling.
# 		denom = 4.0 if single_object_mode == 0 else 1.0

# 		shaped_pre_success_scaled_max = shaped_pre_success_raw_max * reward_scale / denom
# 		sparse_success_scaled = sparse_success_raw_max * reward_scale / denom

# 		# Naive upper bound if success and shaping were added simultaneously.
# 		# In practice success usually dominates and shaping is effectively not the driver.
# 		combined_naive_upper_raw = sparse_success_raw_max + shaped_pre_success_raw_max
# 		combined_naive_upper_scaled = combined_naive_upper_raw * reward_scale / denom

# 		print("\n=== PickPlaceCan numeric reward summary ===")
# 		print(f"reward_scale={reward_scale}, single_object_mode={single_object_mode}")
# 		print(f"staged component maxima (raw): reach={reach_max}, grasp={grasp_max}, lift={lift_max}, hover={hover_max}")
# 		print(f"max pre-success shaped reward (scaled): {shaped_pre_success_scaled_max:.3f}")
# 		print(f"success reward term (scaled): {sparse_success_scaled:.3f}")
# 		print(f"naive combined upper bound success+shaping (scaled): {combined_naive_upper_scaled:.3f}")
# 		print("For Can single-object runs, success reward is typically 1.0 when placed correctly.")

# print("=== PickPlaceCan.reward ===")
# print(inspect.getsource(PickPlaceCan.reward))

# print("\n=== PickPlace.staged_rewards ===")
# print(inspect.getsource(PickPlace.staged_rewards))

# print("\n=== PickPlace._check_success ===")
# print(inspect.getsource(PickPlace._check_success))

# print_can_reward_numeric_summary(reward_scale=1.0, single_object_mode=2)


# import inspect
# from robosuite.environments.manipulation import Lift

# reward_run = inspect.getsource(Lift.reward)

# print("=== Lift.reward ===")
# print(reward_run)


from lerobot.datasets.lerobot_dataset import LeRobotDataset

# This downloads and properly builds the local folder structure
dataset = LeRobotDataset("poolvarine/SARM-robosuite-can-mh-stages")