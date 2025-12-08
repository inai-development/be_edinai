/**
 * Superadministration Portal Login Page
 * Dedicated login page for superadministration portal
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { LogIn, AlertCircle } from 'lucide-react';
import toast from 'react-hot-toast';
import Button from '../../components/ui/Button';
import InputField from '../../components/ui/InputField';

const SuperadministrationLogin = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [credentials, setCredentials] = useState({
    email: '',
    password: '',
  });

  const handleInputChange = (field) => (e) => {
    setCredentials(prev => ({
      ...prev,
      [field]: e.target.value
    }));
    setError('');
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    
    if (!credentials.email || !credentials.password) {
      setError('Please enter both email and password');
      return;
    }

    try {
      setLoading(true);
      const response = await fetch('/api/superadministration/portal/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: credentials.email,
          password: credentials.password,
        }),
      });

      const data = await response.json();

      if (data.status) {
        // Store superadmin token
        localStorage.setItem('superadmin_token', data.data.token);
        localStorage.setItem('superadmin_user', JSON.stringify({
          role: 'superadmin',
          user_type: 'superadmin',
        }));
        
        toast.success('Superadministration portal login successful!');
        navigate('/superadministration/dashboard');
      } else {
        setError(data.message || 'Login failed');
        toast.error(data.message || 'Login failed');
      }
    } catch (err) {
      const errorMsg = err.message || 'An error occurred during login';
      setError(errorMsg);
      toast.error(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-dark-bg via-dark-bg to-dark-bg-secondary flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md"
      >
        <div className="card p-8 shadow-2xl">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="flex justify-center mb-4">
              <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-4 rounded-lg">
                <LogIn className="w-8 h-8 text-white" />
              </div>
            </div>
            <h1 className="text-3xl font-bold text-dark-text-primary mb-2">
              Superadministration Portal
            </h1>
            <p className="text-dark-text-secondary">
              Secure access to system administration
            </p>
          </div>

          {/* Error Alert */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg flex items-start gap-3"
            >
              <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
              <p className="text-red-400 text-sm">{error}</p>
            </motion.div>
          )}

          {/* Login Form */}
          <form onSubmit={handleLogin} className="space-y-6">
            <InputField
              label="Email"
              type="email"
              placeholder="Enter your email"
              value={credentials.email}
              onChange={handleInputChange('email')}
              disabled={loading}
              required
            />

            <InputField
              label="Password"
              type="password"
              placeholder="Enter your password"
              value={credentials.password}
              onChange={handleInputChange('password')}
              disabled={loading}
              required
            />

            <Button
              type="submit"
              variant="primary"
              size="large"
              disabled={loading}
              className="w-full"
            >
              {loading ? 'Logging in...' : 'Login to Portal'}
            </Button>
          </form>

          {/* Footer */}
          <div className="mt-8 pt-6 border-t border-dark-border">
            <p className="text-center text-dark-text-secondary text-sm">
              This portal is restricted to authorized superadministrators only.
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default SuperadministrationLogin;
