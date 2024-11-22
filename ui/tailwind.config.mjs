/** @type {import('tailwindcss').Config} */
export default {
	content: [
		"./index.html",
		"./src/**/*.{js,ts,jsx,tsx}",
	],
	darkMode: "class",  // Changed to just "class" instead of ['class', "class"]
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
			}
		}
	},
	plugins: [require("tailwindcss-animate")],
}