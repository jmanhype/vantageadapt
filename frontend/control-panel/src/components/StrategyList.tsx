import { Stack, Card, Text, Group, Badge, Button, Alert } from "@mantine/core";
import { useState, useEffect } from "react";
import axios from "axios";
import { API_BASE_URL } from "../config";

interface Strategy {
  id: number;
  theme: string;
  created_at: string;
  status: 'active' | 'inactive';
  performance: string;
}

interface Performance {
  total_return: number | null;
  sharpe_ratio: number | null;
}

export default function StrategyList() {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<number | null>(null);

  useEffect(() => {
    fetchStrategies();
    // Set up polling for updates
    const interval = setInterval(fetchStrategies, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchStrategies = async () => {
    try {
      const response = await axios.get<Strategy[]>(`${API_BASE_URL}/api/strategy/list`);
      setStrategies(response.data);
      setError(null);
    } catch (err) {
      console.error('Error:', err);
      setError('Failed to fetch strategies');
    } finally {
      setLoading(false);
    }
  };

  const executeStrategy = async (id: number, live: boolean) => {
    try {
      setActionLoading(id);
      await axios.post(`${API_BASE_URL}/api/strategy/execute`, {
        strategy_id: id,
        data_path: '../big_optimize_1016.pkl',
        live
      });
      await fetchStrategies();
    } catch (err) {
      console.error('Error executing strategy:', err);
      setError('Failed to execute strategy');
    } finally {
      setActionLoading(null);
    }
  };

  const stopStrategy = async (id: number) => {
    try {
      setActionLoading(id);
      await axios.post(`${API_BASE_URL}/api/strategy/stop`, {
        strategy_id: id
      });
      await fetchStrategies();
    } catch (err) {
      console.error('Error stopping strategy:', err);
      setError('Failed to stop strategy');
    } finally {
      setActionLoading(null);
    }
  };

  if (loading) return <Text>Loading strategies...</Text>;
  if (error) return <Alert color="red" title="Error">{error}</Alert>;
  if (!strategies.length) return <Text>No strategies found</Text>;

  return (
    <Stack gap="md">
      {strategies.map((strategy) => {
        const performance: Performance = strategy.performance ? JSON.parse(strategy.performance) : { total_return: null, sharpe_ratio: null };
        const isLoading = actionLoading === strategy.id;
        
        return (
          <Card key={strategy.id} withBorder shadow="sm">
            <Group justify="space-between" mb="xs">
              <Text fw={500}>{strategy.theme}</Text>
              <Badge color={strategy.status === 'active' ? 'green' : 'gray'}>
                {strategy.status}
              </Badge>
            </Group>

            <Group gap="xs" mb="md">
              <Text size="sm">Return:</Text>
              <Text 
                size="sm" 
                fw={500} 
                c={performance.total_return && performance.total_return >= 0 ? 'green' : 'red'}
              >
                {performance.total_return !== null ? `${(performance.total_return * 100).toFixed(2)}%` : 'N/A'}
              </Text>
              <Text size="sm" ml="md">Sharpe:</Text>
              <Text size="sm" fw={500}>
                {performance.sharpe_ratio !== null ? performance.sharpe_ratio.toFixed(2) : 'N/A'}
              </Text>
            </Group>

            <Group>
              {strategy.status === 'inactive' ? (
                <>
                  <Button 
                    size="xs"
                    loading={isLoading}
                    onClick={() => executeStrategy(strategy.id, false)}
                  >
                    Backtest
                  </Button>
                  <Button 
                    size="xs"
                    color="green"
                    loading={isLoading}
                    onClick={() => executeStrategy(strategy.id, true)}
                  >
                    Live Trade
                  </Button>
                </>
              ) : (
                <Button 
                  size="xs"
                  color="red"
                  loading={isLoading}
                  onClick={() => stopStrategy(strategy.id)}
                >
                  Stop
                </Button>
              )}
            </Group>
          </Card>
        );
      })}
    </Stack>
  );
}
