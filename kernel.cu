#include "stdio.h"
#include "stdio.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

extern "C" {
	__global__ void gpuMain(int *dest, int n_players, float prob)
	{
		const int id = blockDim.x*blockIdx.x + threadIdx.x;

		/*Set up random number generator*/
		curandState state;
		curand_init((unsigned long long)clock() + id, 0, 0, &state);

		int plyrs[{{ MAX_PLYRS }}] = { 0 }; /*Creates an array to store player status.
								 Number of players cannot exceed this length.
								 All future references to the number of players 
								 will rely on n_players, NOT the array length.*/

		int survivors = n_players; /*Initial number of survivors*/

		while (survivors > 1) {
			int i = 0;
			for (i = 0; i < n_players; i++) {
				if (plyrs[i] != 1) { /*if player is alive*/
					bool has_killed = false;
					int next_space = 1;
					while (has_killed == false) {
						int next_player_position;
						if (i + next_space >= n_players) {
							next_player_position = i + next_space - n_players;
						}
						else {
							next_player_position = i + next_space;
						}
						if (plyrs[next_player_position] != 1) { /*if next player is alive*/
							float dice_roll = curand_uniform_double(&state);
							if (dice_roll <= prob) {
								plyrs[next_player_position] = 1;
								survivors -= 1;
							}
							has_killed = true;
						}
						else {  /*if next player is dead*/
							next_space += 1;
						}
					}
				}
			}
		}

		/*Print out final array*/
		int j;
		int winner;
		for (j = 0; j < n_players; j++) {
			if (plyrs[j] == 0) {
				winner = j;
			}
		}

		dest[id] = winner;
	}
}