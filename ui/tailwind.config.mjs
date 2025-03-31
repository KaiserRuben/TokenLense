/** @type {import('tailwindcss').Config} */
export default {
	content: [
		"./index.html",
		"./src/**/*.{js,ts,jsx,tsx}",
	],
	darkMode: ["class", "class"],
	theme: {
    	extend: {
    		colors: {
    			background: {
    				light: 'rgba(255, 255, 255, 1)',
    				dark: 'rgb(15, 15, 20)',
    				DEFAULT: 'var(--background)'
    			},
    			surface: {
    				dark: 'rgba(15, 15, 20, 0.95)',
    				light: 'rgba(255, 255, 255, 0.7)'
    			},
    			border: {
    				light: 'rgba(255, 255, 255, 0.1)',
    				dark: 'rgba(0, 0, 0, 0.1)',
    				DEFAULT: 'hsl(var(--border))'
    			},
    			content: {
    				primary: {
    					dark: '#FFFFFF',
    					light: '#1A1A1A'
    				},
    				secondary: {
    					dark: '#94A3B8',
    					light: '#64748B'
    				},
    				muted: {
    					dark: '#64748B',
    					light: '#94A3B8'
    				}
    			},
    			accent: {
    				positive: {
    					primary: '#10B981',
    					subtle: 'rgba(16, 185, 129, 0.2)'
    				},
    				negative: {
    					primary: '#EF4444',
    					subtle: 'rgba(239, 68, 68, 0.2)'
    				},
    				blue: {
    					primary: '#3B82F6',
    					subtle: 'rgba(59, 130, 246, 0.2)'
    				},
    				DEFAULT: 'hsl(var(--accent))',
    				foreground: 'hsl(var(--accent-foreground))'
    			},
    			foreground: 'hsl(var(--foreground))',
    			card: {
    				DEFAULT: 'hsl(var(--card))',
    				foreground: 'hsl(var(--card-foreground))'
    			},
    			popover: {
    				DEFAULT: 'hsl(var(--popover))',
    				foreground: 'hsl(var(--popover-foreground))'
    			},
    			primary: {
    				DEFAULT: 'hsl(var(--primary))',
    				foreground: 'hsl(var(--primary-foreground))'
    			},
    			secondary: {
    				DEFAULT: 'hsl(var(--secondary))',
    				foreground: 'hsl(var(--secondary-foreground))'
    			},
    			muted: {
    				DEFAULT: 'hsl(var(--muted))',
    				foreground: 'hsl(var(--muted-foreground))'
    			},
    			destructive: {
    				DEFAULT: 'hsl(var(--destructive))',
    				foreground: 'hsl(var(--destructive-foreground))'
    			},
    			input: 'hsl(var(--input))',
    			ring: 'hsl(var(--ring))',
    			chart: {
    				'1': 'hsl(var(--chart-1))',
    				'2': 'hsl(var(--chart-2))',
    				'3': 'hsl(var(--chart-3))',
    				'4': 'hsl(var(--chart-4))',
    				'5': 'hsl(var(--chart-5))'
    			}
    		},
    		backdropBlur: {
    			xs: '2px'
    		},
    		transitionDuration: {
    			'400': '400ms'
    		},
    		borderRadius: {
    			xl: '1rem',
    			'2xl': '1.5rem',
    			lg: 'var(--radius)',
    			md: 'calc(var(--radius) - 2px)',
    			sm: 'calc(var(--radius) - 4px)'
    		},
    		boxShadow: {
    			'card-dark': '0 4px 24px -4px rgba(0, 0, 0, 0.3)',
    			'card-light': '0 4px 24px -4px rgba(0, 0, 0, 0.1)',
    			'hover-dark': '0 8px 32px -4px rgba(0, 0, 0, 0.5)',
    			'hover-light': '0 8px 32px -4px rgba(0, 0, 0, 0.2)'
    		},
    		keyframes: {
    			fadeIn: {
    				'0%': {
    					opacity: '0'
    				},
    				'100%': {
    					opacity: '1'
    				}
    			},
    			slideIn: {
    				'0%': {
    					transform: 'translateY(20px)',
    					opacity: '0'
    				},
    				'100%': {
    					transform: 'translateY(0)',
    					opacity: '1'
    				}
    			},
    			slideInLeft: {
    				'0%': {
    					transform: 'translateX(-20px)',
    					opacity: '0'
    				},
    				'100%': {
    					transform: 'translateX(0)',
    					opacity: '1'
    				}
    			},
    			slideInRight: {
    				'0%': {
    					transform: 'translateX(20px)',
    					opacity: '0'
    				},
    				'100%': {
    					transform: 'translateX(0)',
    					opacity: '1'
    				}
    			},
    			scaleIn: {
    				'0%': {
    					transform: 'scale(0.95)',
    					opacity: '0'
    				},
    				'100%': {
    					transform: 'scale(1)',
    					opacity: '1'
    				}
    			},
    			float: {
    				'0%, 100%': {
    					transform: 'translateY(0)'
    				},
    				'50%': {
    					transform: 'translateY(-5px)'
    				}
    			},
    			pulse: {
    				'0%, 100%': {
    					opacity: '1'
    				},
    				'50%': {
    					opacity: '0.5'
    				}
    			},
    			shimmer: {
    				'0%': {
    					backgroundPosition: '-1000px 0'
    				},
    				'100%': {
    					backgroundPosition: '1000px 0'
    				}
    			},
    			expandWidth: {
    				'0%': {
    					width: '0%'
    				},
    				'100%': {
    					width: '100%'
    				}
    			},
    			spin: {
    				'0%': {
    					transform: 'rotate(0deg)'
    				},
    				'100%': {
    					transform: 'rotate(360deg)'
    				}
    			},
    			bounce: {
    				'0%, 100%': {
    					transform: 'translateY(0)'
    				},
    				'50%': {
    					transform: 'translateY(-10px)'
    				}
    			},
    			'accordion-down': {
    				from: {
    					height: '0'
    				},
    				to: {
    					height: 'var(--radix-accordion-content-height)'
    				}
    			},
    			'accordion-up': {
    				from: {
    					height: 'var(--radix-accordion-content-height)'
    				},
    				to: {
    					height: '0'
    				}
    			}
    		},
    		animation: {
    			fadeIn: 'fadeIn 0.5s ease-out',
    			slideIn: 'slideIn 0.5s ease-out forwards',
    			slideInLeft: 'slideInLeft 0.5s ease-out forwards',
    			slideInRight: 'slideInRight 0.5s ease-out forwards',
    			scaleIn: 'scaleIn 0.5s ease-out forwards',
    			float: 'float 3s ease-in-out infinite',
    			pulse: 'pulse 2s ease-in-out infinite',
    			shimmer: 'shimmer 2.5s linear infinite',
    			expandWidth: 'expandWidth 0.5s ease-out forwards',
    			spin: 'spin 1s linear infinite',
    			bounce: 'bounce 2s ease-in-out infinite',
    			'accordion-down': 'accordion-down 0.2s ease-out',
    			'accordion-up': 'accordion-up 0.2s ease-out'
    		},
    		transitionDelay: {
    			'0': '0ms',
    			'100': '100ms',
    			'200': '200ms',
    			'300': '300ms',
    			'400': '400ms',
    			'500': '500ms'
    		}
    	}
    },
	plugins: [
		require("tailwindcss-animate"),
	],
}