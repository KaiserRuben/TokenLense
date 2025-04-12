# TokenLense Visualization Framework

TokenLense is an advanced visualization framework for exploring language model token attribution data. It provides interactive visualizations that help researchers and developers understand how language models make decisions during text generation.

## Features

- **Token Attribution Visualization**: See how tokens influence each other during text generation
- **Method Comparison**: Compare different attribution methods side-by-side
- **Model Comparison**: Compare attribution patterns across different models
- **Context Window Adjustment**: Explore different context sizes for token relationships
- **Performance Metrics**: Analyze attribution method performance and efficiency

## Getting Started

### Prerequisites

- Node.js 18+ and Bun (recommended) or npm
- Access to a TokenLense API instance (locally or remote)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tokenlense.git
   cd tokenlense/visualizer
   ```

2. Install dependencies:
   ```bash
   bun install
   # or with npm
   npm install
   ```

3. Configure the API endpoint:
   - Create a `.env.local` file in the `visualizer` directory
   - Add `NEXT_PUBLIC_API_URL=http://your-api-endpoint` (default: http://localhost:8000)

4. Start the development server:
   ```bash
   bun run dev
   # or with npm
   npm run dev
   ```

5. Open your browser to http://localhost:3000

### Building for Production

```bash
bun run build
# or with npm
npm run build
```

## Project Structure

- `/app`: Next.js application pages and routes
- `/components`: React components
  - `/attribution`: Attribution visualization components
  - `/layout`: Layout components like Header and Footer
- `/lib`: Utility functions and API client
- `/docs`: Project documentation

## Usage

1. Select a model from the homepage
2. Choose an attribution method for the model
3. Select an attribution file to visualize
4. Explore token relationships using the visualization tools
5. Adjust settings like context window size and aggregation method
6. Compare different methods and models

## Development

To contribute to the development of TokenLense:

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Implement your changes
3. Add tests if appropriate
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Built with Next.js 15, React 19, and Tailwind CSS
- Uses D3.js for data visualization
- Based on InSeq attribution data format

## Contact

For questions or support, please open an issue on the GitHub repository.